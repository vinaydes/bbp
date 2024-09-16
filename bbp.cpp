/*
      inf         /                                             \
      ---    1    |    4            2          1           1    |
      \    -----  | --------  -  -------- - -------- - -------- |
pi =  /     16^k  |  8k + 1       8k + 4     8k + 5     8k + 6  |
      ---         \                                             /
    k = 0

 */

#include <cstdio>
#include <cstdint>
#include <math.h>
#include <iostream>
#include <gmp.h>

// return (a * (16 ^ exp)) % m
int pow16mod(int exp, int a, int m) {
  int64_t base = 16; // We use uint64_t to avoid overflow in intermediate calculations
  int64_t result = 1;

  if (m == 1) return base*a;

  while (exp > 0) {
    if (exp & 1) {
      result = (result * base) % m;
    }
    base = (base * base) % m;
    exp >>= 1;
  }
  result = (result * a) % m;
  return static_cast<int>(result);
}

// Accumulate real number between [0, 1) with fixed precision
// The precision is sizeof(IType) * 8 * WORDS bits
// For example if IType = uint32_t and WORDS = 2
// then number of bits = 4 * 8 * 2 = 64 bits,
// and the accumulator holds the sum * 2^64
// Integer part of the accumulation is completely ignored
template <typename IType, typename WType, size_t WORDS>
class accumulator {
  public:
    IType w[WORDS];
    constexpr static int i_width = sizeof(IType) * 8;
    mpf_t r;

    accumulator() {
      mpf_set_default_prec(500);
      reset();
      mpf_init(r);
    }

    ~accumulator() {
      //mpf_clear(r);
    }

    void reset() {
      for (size_t i = 0; i < WORDS; i++) {
        w[i] = IType(0);
      }
      mpf_set_d(r, 0.0);
    }

    void adjust(bool debug=false) {
      if (debug) {
        mpf_out_str(stdout, 10, 10, r);
      }
      mpz_t ti;
      mpf_t tf;

      mpz_init(ti);
      mpf_init(tf);

      mpz_set_f(ti, r);
      mpf_set_z(tf, ti);
      mpf_sub(r, r, tf);
      mpf_clear(tf);
      mpz_clear(ti);
    }

    bool is_zero() {
      for (size_t i = 0; i < WORDS; i++) {
        if (w[i] != IType(0))
          return false;
      }
      return true;
    }

    void add(IType numerator, IType denominator) {
      mpf_mul_ui(r, r, uint64_t(denominator));
      mpf_add_ui(r, r, uint64_t(numerator));
      mpf_div_ui(r, r, uint64_t(denominator));
      adjust();
      IType op[WORDS];
      for (size_t i = 0; i < WORDS; i++) {
        WType temp = WType(numerator) << i_width;
        op[i] = IType(temp / denominator);
        numerator = IType(temp - op[i] * denominator);
      }
      IType carry;
      // Rounding logic to improve accuracy with same precision
      if (numerator >= denominator/2) {
        carry = 1;
      } else {
        carry = 0;
      }
      for (size_t i = WORDS; i > 0; i--) {
        w[i - 1] += (op[i - 1]);

        if (w[i - 1] < op[i - 1]) {
          w[i - 1] += carry;
          carry = 1;
        } else {
          w[i - 1] += carry;
          if (w[i - 1] < carry) {
            carry = 1;
          } else {
            carry = 0;
          }
        }
      }
    }

    void add(accumulator<IType, WType, WORDS>& op1) {
      IType carry = 0;
      for (size_t i = WORDS - 1; ; i--) {
        w[i] += (op1.w[i] + carry);
        if (w[i] <= op1.w[i])
          carry = 1;
        else
          carry = 0;
        if (i == 0) break;
      }
      mpf_add(r, r, op1.r);
      adjust();
    }

    void rshift(int n) {
      if (n < 0) {
        printf("Negative amount for shift is not allowed\n");
        return;
      }

      if (n >= (WORDS * i_width)) {
        reset();
        return;
      }

      int bit_shift = n % i_width;
      int word_shift = n / i_width;
      if (bit_shift != 0) {
        for (size_t i = WORDS - 1; i >= 1; i--) {
          w[i] = (w[i - 1] << (i_width - bit_shift)) | (w[i] >> bit_shift);
        }
      }
      w[0] = w[0] >> bit_shift;

      for (size_t i = WORDS - 1; ; i--) {
        if (i >= word_shift) {
          w[i] = w[i - word_shift];
        } else {
          w[i] = 0;
        }
        if (i == 0) break;
      }
      mpf_div_2exp(r, r, n);
      adjust();
    }

    void lshift(int n) {
      if (n < 0) {
        printf("Negative amount for shift is not allowed\n");
        return;
      }

      if (n >= (WORDS * i_width)) {
        reset();
        return;
      }

      int bit_shift = n % i_width;
      int word_shift = n / i_width;
      if (bit_shift != 0) {
        for (size_t i = 0; i < WORDS - 1; i++) {
          w[i] = (w[i] << bit_shift) | (w[i + 1] >> (i_width  - bit_shift));
        }
      }
      w[WORDS - 1] = w[WORDS - 1] << bit_shift;

      for (size_t i = 0; i < WORDS; i++) {
        if (i + word_shift < WORDS) {
          w[i] = w[i + word_shift];
        } else {
          w[i] = 0;
        }
      }
      mpf_mul_2exp(r, r, n);
      adjust();
    }

    void print() {
      mpf_t a;
      mpf_t s;

      mpf_init(a);
      mpf_init(s);

      mpf_set_ui(s, 0);
      mpf_set_ui(a, 0);

      for (size_t i = 0; i < WORDS; i++) {
        mpf_set_ui(a, uint64_t(w[i]));
        mpf_div_2exp(a, a, (i + 1) * i_width);
        mpf_add(s, s, a);
      }

      mpf_t diff;
      mpf_init(diff);
      mpf_sub(diff, r, s);
      mpf_out_str(stdout, 10, 50, r);
      printf("\t");
      mpf_out_str(stdout, 10, 5, diff);
      printf("\n");
      mpf_clear(diff);
      mpf_clear(s);
      mpf_clear(a);
    }

    double to_real() {
      double ret = 0.0;
      for (size_t i = 0; i < WORDS; i++) {
        ret += ldexp(double(w[i]), -32*(i+1));
      }
      return ret;
    }

    void decimal_expansion() {

      constexpr size_t DIGITS = 2 * size_t(i_width * WORDS * 0.301029995663981195213738894724493026); 
      uint8_t decimal[DIGITS]; // This will hold decimal expansion, number of digits = DIGITS
      uint8_t fraction[DIGITS];
      // initialize the expansion to 0.5
      decimal[0] = 5;
      fraction[0] = 0;
      for (size_t i = 1; i < DIGITS; i++) {
        decimal[i] = 0;
        fraction[i] = 0;
      }

      for (size_t i = 0; i < WORDS; i++) {
        for (size_t b = i_width; b > 0; b--) {

          if ((w[i] >> (b - 1)) & 0x1) {
            uint8_t carry = 0;
            for (size_t j = DIGITS; j > 0; j--) {
              fraction[j - 1] += (decimal[j - 1] + carry);
              carry = fraction[j - 1] / 10;
              fraction[j - 1] %= 10;
            }
          }

          uint8_t q = 0;
          uint8_t r = 0;
          for (size_t j = 0; j < DIGITS; j++) {
            q = (r*10 + decimal[j]) / 2;
            r = (r*10 + decimal[j]) - 2 * q;
            decimal[j] = q;
          }
        }
      }
      printf("0.");
      for (size_t j = 0; j < DIGITS; j++) {
        printf("%c", '0' + fraction[j]);
      }
      printf("\n");

      return;
    }
};

using acc_type = accumulator<uint32_t, uint64_t, 4>;

// Calculate fractional part of 16^d * S_j where 
// S_j = sum ( 1 / ( (16^k) * (8*k + j) ) ) for k = 0 to inf
// Equation 5, 6, 7 from https://www.davidhbailey.com/dhbpapers/bbp-alg.pdf
acc_type calculate_sum(uint32_t d, uint32_t j) {
  
  acc_type acc;
  for (int k = 0; k <= d; k++) {
    uint32_t denominator = 8*k + j;
    uint32_t numerator = pow16mod(d - k, 1, denominator);
    acc.add(numerator, denominator);
  }
  //acc.print();
  for (uint32_t k = d + 1; k < d + 300; k++) {
    acc_type term;
    term.add(uint32_t(1), 8*k + j);
    term.rshift((k - d)*4);
    if (term.is_zero()) {
      printf("Breaking at %u\n", k);
      break;
    }
    acc.add(term);
  }

  return acc;
}

int main()
{
  acc_type pi_16d;
  acc_type term;
  uint32_t d = 1000000;
  term = calculate_sum(d, 6);
  term.print();


  term.decimal_expansion();
  /*acc_type term1;

  term1.add(uint32_t(1), 8000065);
  term1.print();
  term1.rshift(32);
  term1.print();*/
  //term.decimal_expansion();

  return 0;
}


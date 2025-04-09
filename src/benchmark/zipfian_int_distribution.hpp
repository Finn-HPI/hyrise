/*
 * MIT License
 *
 * Copyright (c) 2017 Lucas Lersch
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* Implementation derived from:
 * "Quickly Generating Billion-Record Synthetic Databases", Jim Gray et al,
 * SIGMOD 1994
 */

/*
 * The zipfian_int_distribution class is intended to be compatible with other
 * distributions introduced in #include <random> by the C++11 standard.
 * 
 * Usage example:
 * #include <random>
 * #include "zipfian_int_distribution.h"
 * int main()
 * {
 *   std::default_random_engine generator;
 *   zipfian_int_distribution<int> distribution(1, 10, 0.99);
 *   int i = distribution(generator);
 * }
 */

/*
 * IMPORTANT: constructing the distribution object requires calculating the zeta
 * value which becomes prohibetively expensive for very large ranges. As an
 * alternative for such cases, the user can pass the pre-calculated values and
 * avoid the calculation every time.
 * 
 * Usage example:
 * #include <random>
 * #include "zipfian_int_distribution.h"
 * int main()
 * {
 *   std::default_random_engine generator;
 *   zipfian_int_distribution<int>::param_type p(1, 1e6, 0.99, 27.000);
 *   zipfian_int_distribution<int> distribution(p);
 *   int i = distribution(generator);
 * }
 */

#include <cassert>
#include <cmath>
#include <limits>
#include <random>

template <typename IntType = int>
class zipfian_int_distribution {
  static_assert(std::is_integral<IntType>::value, "Template argument not an integral type.");

 public:
  /** The type of the range of the distribution. */
  typedef IntType result_type;

  /** Parameter type. */
  struct param_type {
    typedef zipfian_int_distribution<IntType> distribution_type;

    explicit param_type(IntType _a = 0, IntType _b = std::numeric_limits<IntType>::max(), double _theta = 0.99)
        : _m_a(_a), _m_b(_b), _m_theta(_theta), _m_zeta(zeta(_m_b - _m_a + 1, _theta)), _m_zeta2theta(zeta(2, _theta)) {
      assert(_m_a <= _m_b && _m_theta > 0.0 && _m_theta < 1.0);
    }

    explicit param_type(IntType _a, IntType _b, double _theta, double _zeta)
        : _m_a(_a), _m_b(_b), _m_theta(_theta), _m_zeta(_zeta), _m_zeta2theta(zeta(2, _theta)) {
      __glibcxx_assert(_m_a <= _m_b && _m_theta > 0.0 && _m_theta < 1.0);
    }

    result_type a() const {
      return _m_a;
    }

    result_type b() const {
      return _m_b;
    }

    double theta() const {
      return _m_theta;
    }

    double zeta() const {
      return _m_zeta;
    }

    double zeta2theta() const {
      return _m_zeta2theta;
    }

    friend bool operator==(const param_type& _p1, const param_type& _p2) {
      return _p1._m_a == _p2._m_a && _p1._m_b == _p2._m_b && _p1._m_theta == _p2._m_theta &&
             _p1._m_zeta == _p2._m_zeta && _p1._m_zeta2theta == _p2._m_zeta2theta;
    }

   private:
    IntType _m_a;
    IntType _m_b;
    double _m_theta;
    double _m_zeta;
    double _m_zeta2theta;

    /**
     * @brief Calculates zeta.
     *
     * @param __n [IN]  The size of the domain.
     * @param __theta [IN]  The skew factor of the distribution.
     */
    double zeta(unsigned long _n, double _theta) {
      double ans = 0.0;
      for (unsigned long i = 1; i <= _n; ++i)
        ans += std::pow(1.0 / i, _theta);
      return ans;
    }
  };

 public:
  /**
   * @brief Constructs a zipfian_int_distribution object.
   *
   * @param __a [IN]  The lower bound of the distribution.
   * @param __b [IN]  The upper bound of the distribution.
   * @param __theta [IN]  The skew factor of the distribution.
   */
  explicit zipfian_int_distribution(IntType _a = IntType(0), IntType _b = IntType(1), double _theta = 0.99)
      : _m_param(_a, _b, _theta) {}

  explicit zipfian_int_distribution(const param_type& _p) : _m_param(_p) {}

  /**
   * @brief Resets the distribution state.
   *
   * Does nothing for the zipfian int distribution.
   */
  void reset() {}

  result_type a() const {
    return _m_param.a();
  }

  result_type b() const {
    return _m_param.b();
  }

  double theta() const {
    return _m_param.theta();
  }

  /**
   * @brief Returns the parameter set of the distribution.
   */
  param_type param() const {
    return _m_param;
  }

  /**
   * @brief Sets the parameter set of the distribution.
   * @param __param The new parameter set of the distribution.
   */
  void param(const param_type& _param) {
    _m_param = _param;
  }

  /**
   * @brief Returns the inclusive lower bound of the distribution range.
   */
  result_type min() const {
    return this->a();
  }

  /**
   * @brief Returns the inclusive upper bound of the distribution range.
   */
  result_type max() const {
    return this->b();
  }

  /**
   * @brief Generating functions.
   */
  template <typename UniformRandomNumberGenerator>
  result_type operator()(UniformRandomNumberGenerator& _urng) {
    return this->operator()(_urng, _m_param);
  }

  template <typename UniformRandomNumberGenerator>
  result_type operator()(UniformRandomNumberGenerator& _urng, const param_type& _p) {
    double alpha = 1 / (1 - _p.theta());
    double eta = (1 - std::pow(2.0 / (_p.b() - _p.a() + 1), 1 - _p.theta())) / (1 - _p.zeta2theta() / _p.zeta());

    double u =
        std::generate_canonical<double, std::numeric_limits<double>::digits, UniformRandomNumberGenerator>(_urng);

    double uz = u * _p.zeta();
    if (uz < 1.0)
      return _p.a();
    if (uz < 1.0 + std::pow(0.5, _p.theta()))
      return _p.a() + 1;

    return static_cast<result_type>(_p.a() + ((_p.b() - _p.a() + 1) * std::pow((eta * u) - eta + 1, alpha)));
  }

  /**
   * @brief Return true if two zipfian int distributions have
   *        the same parameters.
   */
  friend bool operator==(const zipfian_int_distribution& _d1, const zipfian_int_distribution& _d2) {
    return _d1._m_param == _d2._m_param;
  }

 private:
  param_type _m_param;
};

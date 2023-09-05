#include <unittest/unittest.h>

#include <thrust/complex.h>
#include <thrust/detail/config.h>

#include <complex>
#include <iostream>
#include <sstream>
#include <type_traits>

/* 
   The following tests do not check for the numerical accuracy of the operations.
   That is tested in a separate program (complex_accuracy.cpp) which requires mpfr, 
   and takes a lot of time to run.   
 */

template<typename T>
struct TestComplexSizeAndAlignment
{
  void operator()()
  {
    THRUST_STATIC_ASSERT(
      sizeof(thrust::complex<T>) == sizeof(T) * 2
    );
    THRUST_STATIC_ASSERT(
      THRUST_ALIGNOF(thrust::complex<T>) == THRUST_ALIGNOF(T) * 2
    );

    THRUST_STATIC_ASSERT(
      sizeof(thrust::complex<T const>) == sizeof(T) * 2
    );
    THRUST_STATIC_ASSERT(
      THRUST_ALIGNOF(thrust::complex<T const>) == THRUST_ALIGNOF(T) * 2
    );
  }
};
SimpleUnitTest<TestComplexSizeAndAlignment, FloatingPointTypes> TestComplexSizeAndAlignmentInstance;

template<typename T>
struct TestComplexConstructors
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(2);
    
    thrust::complex<T> a(data[0],data[1]);
    thrust::complex<T> b(a);
    a = thrust::complex<T>(data[0],data[1]);
    ASSERT_ALMOST_EQUAL(a,b);
    
    a = thrust::complex<T>(data[0]);
    ASSERT_EQUAL(data[0], a.real());
    ASSERT_EQUAL(T(0), a.imag());
    
    a = thrust::complex<T>();
    ASSERT_ALMOST_EQUAL(a,std::complex<T>(0));
    
    a = thrust::complex<T>(thrust::complex<float>(static_cast<float>(data[0]),static_cast<float>(data[1])));
    ASSERT_ALMOST_EQUAL(a,b);
    
    a = thrust::complex<T>(thrust::complex<double>(static_cast<double>(data[0]),static_cast<double>(data[1])));
    ASSERT_ALMOST_EQUAL(a,b);
    
    a = thrust::complex<T>(std::complex<float>(static_cast<float>(data[0]),static_cast<float>(data[1])));
    ASSERT_ALMOST_EQUAL(a,b);
    
    a = thrust::complex<T>(std::complex<double>(static_cast<double>(data[0]),static_cast<double>(data[1])));
    ASSERT_ALMOST_EQUAL(a,b);
  }
};
SimpleUnitTest<TestComplexConstructors, FloatingPointTypes> TestComplexConstructorsInstance;


template<typename T>
struct TestComplexGetters
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(2);

    thrust::complex<T> z(data[0], data[1]);

    ASSERT_EQUAL(data[0], z.real());
    ASSERT_EQUAL(data[1], z.imag());

    z.real(data[1]);
    z.imag(data[0]);
    ASSERT_EQUAL(data[1], z.real());
    ASSERT_EQUAL(data[0], z.imag());

    volatile thrust::complex<T> v(data[0], data[1]);

    ASSERT_EQUAL(data[0], v.real());
    ASSERT_EQUAL(data[1], v.imag());

    v.real(data[1]);
    v.imag(data[0]);
    ASSERT_EQUAL(data[1], v.real());
    ASSERT_EQUAL(data[0], v.imag());
  }
};
SimpleUnitTest<TestComplexGetters, FloatingPointTypes> TestComplexGettersInstance;

template<typename T>
struct TestComplexMemberOperators
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);
    thrust::host_vector<T> data_b = unittest::random_samples<T>(2);

    thrust::complex<T> a(data_a[0], data_a[1]);
    thrust::complex<T> b(data_b[0], data_b[1]);

    std::complex<T> c(a);
    std::complex<T> d(b);

    a += b;
    c += d;
    ASSERT_ALMOST_EQUAL(a,c);

    a -= b;
    c -= d;
    ASSERT_ALMOST_EQUAL(a,c);

    a *= b;
    c *= d;
    ASSERT_ALMOST_EQUAL(a,c);

    a /= b;
    c /= d;
    ASSERT_ALMOST_EQUAL(a,c);

    // casting operator
    c = (std::complex<T>)a;
  }
};
SimpleUnitTest<TestComplexMemberOperators, FloatingPointTypes> TestComplexMemberOperatorsInstance;


template<typename T>
struct TestComplexBasicArithmetic
{
  void operator()(void)
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(2);

    thrust::complex<T> a(data[0], data[1]);
    std::complex<T> b(a);

    // Test the basic arithmetic functions against std
    
    ASSERT_ALMOST_EQUAL(abs(a),abs(b));

    ASSERT_ALMOST_EQUAL(arg(a),arg(b));

    ASSERT_ALMOST_EQUAL(norm(a),norm(b));

    ASSERT_EQUAL(conj(a),conj(b));

    ASSERT_ALMOST_EQUAL(thrust::polar(data[0],data[1]),std::polar(data[0],data[1]));

    // random_samples does not seem to produce infinities so proj(z) == z
    ASSERT_EQUAL(proj(a),a);
    
  }
};
SimpleUnitTest<TestComplexBasicArithmetic, FloatingPointTypes> TestComplexBasicArithmeticInstance;


template<typename T>
struct TestComplexBinaryArithmetic
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);
    thrust::host_vector<T> data_b = unittest::random_samples<T>(2);

    thrust::complex<T> a(data_a[0], data_a[1]);
    thrust::complex<T> b(data_b[0], data_b[1]);

    ASSERT_ALMOST_EQUAL(a * b, std::complex<T>(a) * std::complex<T>(b));
    ASSERT_ALMOST_EQUAL(a * data_b[0], std::complex<T>(a) * data_b[0]);
    ASSERT_ALMOST_EQUAL(data_a[0] * b, data_b[0] * std::complex<T>(b));

    ASSERT_ALMOST_EQUAL(a / b, std::complex<T>(a) / std::complex<T>(b));
    ASSERT_ALMOST_EQUAL(a / data_b[0], std::complex<T>(a) / data_b[0]);
    ASSERT_ALMOST_EQUAL(data_a[0] / b, data_b[0] / std::complex<T>(b));

    ASSERT_EQUAL(a + b, std::complex<T>(a) + std::complex<T>(b));
    ASSERT_EQUAL(a + data_b[0], std::complex<T>(a) + data_b[0]);
    ASSERT_EQUAL(data_a[0] + b, data_b[0] + std::complex<T>(b));

    ASSERT_EQUAL(a - b, std::complex<T>(a) - std::complex<T>(b));
    ASSERT_EQUAL(a - data_b[0], std::complex<T>(a) - data_b[0]);
    ASSERT_EQUAL(data_a[0] - b, data_b[0] - std::complex<T>(b));
  }
};
SimpleUnitTest<TestComplexBinaryArithmetic, FloatingPointTypes> TestComplexBinaryArithmeticInstance;




template<typename T>
struct TestComplexUnaryArithmetic
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);

    thrust::complex<T> a(data_a[0], data_a[1]);

    ASSERT_EQUAL(+a,+std::complex<T>(a));
    ASSERT_EQUAL(-a,-std::complex<T>(a));
    
  }
};
SimpleUnitTest<TestComplexUnaryArithmetic, FloatingPointTypes> TestComplexUnaryArithmeticInstance;


template<typename T>
struct TestComplexExponentialFunctions
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);

    thrust::complex<T> a(data_a[0], data_a[1]);
    std::complex<T> b(a);

    ASSERT_ALMOST_EQUAL(exp(a),exp(b));
    ASSERT_ALMOST_EQUAL(log(a),log(b));
    ASSERT_ALMOST_EQUAL(log10(a),log10(b));
    
  }
};
SimpleUnitTest<TestComplexExponentialFunctions, FloatingPointTypes> TestComplexExponentialFunctionsInstance;


template<typename T>
struct TestComplexPowerFunctions
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);
    thrust::host_vector<T> data_b = unittest::random_samples<T>(2);

    thrust::complex<T> a(data_a[0], data_a[1]);
    thrust::complex<T> b(data_b[0], data_b[1]);
    std::complex<T> c(a);
    std::complex<T> d(b);

    ASSERT_ALMOST_EQUAL(pow(a,b),pow(c,d));
    ASSERT_ALMOST_EQUAL(pow(a,b.real()),pow(c,d.real()));
    ASSERT_ALMOST_EQUAL(pow(a.real(),b),pow(c.real(),d));

    ASSERT_ALMOST_EQUAL(sqrt(a),sqrt(c));

  }
};
SimpleUnitTest<TestComplexPowerFunctions, FloatingPointTypes> TestComplexPowerFunctionsInstance;

template<typename T>
struct TestComplexTrigonometricFunctions
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);

    thrust::complex<T> a(data_a[0], data_a[1]);
    std::complex<T> c(a);

    ASSERT_ALMOST_EQUAL(cos(a),cos(c));
    ASSERT_ALMOST_EQUAL(sin(a),sin(c));
    ASSERT_ALMOST_EQUAL(tan(a),tan(c));

    ASSERT_ALMOST_EQUAL(cosh(a),cosh(c));
    ASSERT_ALMOST_EQUAL(sinh(a),sinh(c));
    ASSERT_ALMOST_EQUAL(tanh(a),tanh(c));

#if THRUST_CPP_DIALECT >= 2011

    ASSERT_ALMOST_EQUAL(acos(a),acos(c));
    ASSERT_ALMOST_EQUAL(asin(a),asin(c));
    ASSERT_ALMOST_EQUAL(atan(a),atan(c));

    ASSERT_ALMOST_EQUAL(acosh(a),acosh(c));
    ASSERT_ALMOST_EQUAL(asinh(a),asinh(c));
    ASSERT_ALMOST_EQUAL(atanh(a),atanh(c));

#endif


  }
};
SimpleUnitTest<TestComplexTrigonometricFunctions, FloatingPointTypes> TestComplexTrigonometricFunctionsInstance;

template<typename T>
struct TestComplexStreamOperators
{
  void operator()(void)
  {
    thrust::host_vector<T> data_a = unittest::random_samples<T>(2);
    thrust::complex<T> a(data_a[0], data_a[1]);
    std::stringstream out;
    out << a;
    thrust::complex<T> b;
    out >> b;
    ASSERT_ALMOST_EQUAL(a,b);
  }
};
SimpleUnitTest<TestComplexStreamOperators, FloatingPointTypes> TestComplexStreamOperatorsInstance;

#if THRUST_CPP_DIALECT >= 2011
template<typename T>
struct TestComplexStdComplexDeviceInterop
{
  void operator()()
  {
    thrust::host_vector<T> data = unittest::random_samples<T>(6);
    std::vector<std::complex<T> > vec(10);
    vec[0] = std::complex<T>(data[0], data[1]);
    vec[1] = std::complex<T>(data[2], data[3]);
    vec[2] = std::complex<T>(data[4], data[5]);

    thrust::device_vector<thrust::complex<T> > device_vec = vec;
    ASSERT_ALMOST_EQUAL(vec[0].real(), thrust::complex<T>(device_vec[0]).real());
    ASSERT_ALMOST_EQUAL(vec[0].imag(), thrust::complex<T>(device_vec[0]).imag());
    ASSERT_ALMOST_EQUAL(vec[1].real(), thrust::complex<T>(device_vec[1]).real());
    ASSERT_ALMOST_EQUAL(vec[1].imag(), thrust::complex<T>(device_vec[1]).imag());
    ASSERT_ALMOST_EQUAL(vec[2].real(), thrust::complex<T>(device_vec[2]).real());
    ASSERT_ALMOST_EQUAL(vec[2].imag(), thrust::complex<T>(device_vec[2]).imag());
  }
};
SimpleUnitTest<TestComplexStdComplexDeviceInterop, FloatingPointTypes> TestComplexStdComplexDeviceInteropInstance;
#endif


template<typename TypeList>
struct TestComplexAllMembersWithPromoting
{
  void operator()(void)
  {
    typedef unittest::get_type_t<TypeList,0> T1;
    typedef unittest::get_type_t<TypeList,1> T2;

    thrust::host_vector<T1> data_a = unittest::random_samples<T1>(2);
    thrust::host_vector<T2> data_b = unittest::random_samples<T2>(4);

    const T1 a = data_a[0];
    const T1 b = data_a[1];
    const T2 c = data_b[2];
    const T2 d = data_b[3];

    const thrust::complex<T1> elem(a, b);
    const std::complex<T1> std_elem = elem;

    ASSERT_ALMOST_EQUAL(elem, elem);
    ASSERT_ALMOST_EQUAL(std_elem, std_elem);

    ASSERT_EQUAL(elem.real(), a);
    ASSERT_EQUAL(elem.imag(), b);

    // copy ctor
    {
      thrust::complex<T2> elem2(elem);
      ASSERT_EQUAL(elem2.real(), (T2) a);
      ASSERT_EQUAL(elem2.imag(), (T2) b);
    }

    // copy ctor with std::complex
    {
      thrust::complex<T2> elem2(std_elem);
      ASSERT_EQUAL(elem2.real(), (T2) a);
      ASSERT_EQUAL(elem2.imag(), (T2) b);
    }

    // assignment from real number
    {
      thrust::complex<T2> elem2 = a;
      ASSERT_EQUAL(elem2.real(), (T2) a);
      ASSERT_EQUAL(elem2.imag(), 0.0F);
    }
    
    // asignment with std::complex;
    {
      thrust::complex<T1> elem2 = std_elem;
      ASSERT_EQUAL(elem2, std_elem);
    }

    // asignment with std::complex, other T
    {
      thrust::complex<T2> elem2 = std_elem;
      ASSERT_EQUAL(elem2, std_elem);
    }

    // no conversion from complex<float> to complex<double>, rest works
    //{
    //  thrust::complex<T2> elem2 = elem;
    //  ASSERT_EQUAL(elem2.real(), a);
    //  ASSERT_EQUAL(elem2.imag(), b);
    //}

    // assignment add
    {
      thrust::complex<T2> elem2(c, d);
      elem2 += elem;
      ASSERT_EQUAL(elem2.real(), a+c);
      ASSERT_EQUAL(elem2.imag(), b+d);
    }

    // assignment substraction
    {
      thrust::complex<T2> elem2(c, d);
      elem2 -= elem;
      ASSERT_EQUAL(elem2.real(), c-a);
      ASSERT_EQUAL(elem2.imag(), d-b);
    }

    // assignment multiplication
    {
      thrust::complex<T2> elem2(c, d);
      elem2 *= elem;
      ASSERT_ALMOST_EQUAL(elem2.real(), (a*c - b*d));
      ASSERT_ALMOST_EQUAL(elem2.imag(), (a*d + b*c));
    }

    // assignment division
    {
      thrust::complex<T2> elem2(c, d);
      elem2 /= elem;
      ASSERT_ALMOST_EQUAL(elem2.real(), (a*c + b*d)/(a*a + b*b));
      ASSERT_ALMOST_EQUAL(elem2.imag(), (a*d - b*c)/(a*a + b*b));
    }

    // assignment add with real
    {
      thrust::complex<T2> elem2(c, d);
      elem2 += a;
      ASSERT_EQUAL(elem2.real(), c+a);
    }

    // assignment substraction with real
    {
      thrust::complex<T2> elem2(c, d);
      elem2 -= a;
      ASSERT_EQUAL(elem2.real(), c-a);
    }

    // assignment multiplication with real
    {
      thrust::complex<T2> elem2(c, d);
      elem2 *= a;
      ASSERT_ALMOST_EQUAL(elem2.real(), (a*c));
      ASSERT_ALMOST_EQUAL(elem2.imag(), (a*d));
    }

    // assignment division with real
    {
      thrust::complex<T2> elem2(c, d);
      elem2 /= a;
      ASSERT_ALMOST_EQUAL(elem2.real(), (a*c)/(a*a));
      ASSERT_ALMOST_EQUAL(elem2.imag(), (a*d)/(a*a));
    }

    // use .real() and .imag() to change real and imag
    {
      thrust::complex<T1> elem2(T1{0.0}, T1{0.0});
      elem2.real(static_cast<T1>(a));
      ASSERT_EQUAL(elem2.real(), static_cast<T1>(a));

      elem2.imag(static_cast<T1>(b));
      ASSERT_EQUAL(elem2.imag(), static_cast<T1>(b));
    }

    // comparision operators
    ASSERT_EQUAL(thrust::complex<T1>(a, b) == thrust::complex<T1>(a, b), true);
    ASSERT_EQUAL(thrust::complex<T1>(a, b) == thrust::complex<T2>(a, b), true);
    ASSERT_EQUAL(thrust::complex<T1>(a, 0.0) == a, true);
    //ASSERT_EQUAL(thrust::complex<T1>(a, 0.0) == static_cast<T2>(a), true);
    ASSERT_EQUAL(a == thrust::complex<T1>(a, 0.0), true);
    //ASSERT_EQUAL(static_cast<T2>(a) == thrust::complex<T1>(a, 0), true);
    ASSERT_EQUAL(std::complex<T1>(a, b) == thrust::complex<T2>(a, b), true);
    ASSERT_EQUAL(thrust::complex<T1>(a, b) == std::complex<T2>(a, b), true);


    ASSERT_EQUAL(thrust::complex<T1>(a, b) != thrust::complex<T1>(c, d), true);
    ASSERT_EQUAL(thrust::complex<T1>(a, b) != thrust::complex<T2>(c, d), true);
    ASSERT_EQUAL(thrust::complex<T1>(a, 0) != b, true);
    //ASSERT_EQUAL(thrust::complex<T1>(a, 0) != c, true);
    ASSERT_EQUAL(b != thrust::complex<T1>(a, 0), true);
    //ASSERT_EQUAL(c != thrust::complex<T1>(a, 0), true);
    ASSERT_EQUAL(std::complex<T1>(a, b) != thrust::complex<T2>(c, d), true);
    ASSERT_EQUAL(thrust::complex<T1>(a, b) != std::complex<T2>(c, d), true);

    // abs
    ASSERT_ALMOST_EQUAL(thrust::abs(elem), std::abs(std_elem));

    // arg
    ASSERT_ALMOST_EQUAL(thrust::arg(elem), std::arg(std_elem));

    // norm
    ASSERT_ALMOST_EQUAL(thrust::norm(elem), std::norm(std_elem));

    // conj
    ASSERT_ALMOST_EQUAL(thrust::conj(elem), std::conj(std_elem));

    // polar, does not compile, cos/sin/... missing
    ASSERT_ALMOST_EQUAL(thrust::polar(a), std::polar(a));
    ASSERT_ALMOST_EQUAL(thrust::polar(a, b), std::polar(a, b));
    //ASSERT_ALMOST_EQUAL(thrust::polar(a, c), std::polar(a, c));

    // proj
    ASSERT_ALMOST_EQUAL(thrust::proj(elem), std::proj(std_elem));

    // add
    auto result = thrust::complex<T1>(a, b) + thrust::complex<T2>(c, d);
    ASSERT_EQUAL(result.real(), a + c);
    ASSERT_EQUAL(result.imag(), b + d); 

    // add with real, does not compile
    //cresult = thrust::complex<T1>(a, b) + c;
    //ASSERT_EQUAL(cresult.real(), a + c);
    //ASSERT_EQUAL(cresult.imag(), b); 

    //cresult = a + thrust::complex<T2>(c, d);
    //ASSERT_EQUAL(cresult.real(), a + c);
    //ASSERT_EQUAL(cresult.imag(), d); 

    // substraction
    result = thrust::complex<T1>(a, b) - thrust::complex<T2>(c, d);
    ASSERT_EQUAL(result.real(), a - c);
    ASSERT_EQUAL(result.imag(), b - d);

    // substraction with real, does not compile
    //cresult = thrust::complex<T1>(a, b) - c;
    //ASSERT_EQUAL(cresult.real(), a - c);
    //ASSERT_EQUAL(cresult.imag(), b); 

    //cresult = a - thrust::complex<T2>(c, d);
    //ASSERT_EQUAL(cresult.real(), a - c);
    //ASSERT_EQUAL(cresult.imag(), d); 

    // multiplication, does not compile for promotion case
    result = thrust::complex<T1>(a, b) * thrust::complex<T2>(c, d);
    ASSERT_ALMOST_EQUAL(result.real(), (a*c - b*d));
    ASSERT_ALMOST_EQUAL(result.imag(), (a*d + b*c));

    // multiplication with real, does not compile
    //cresult = thrust::complex<T1>(a, b) * c;
    //ASSERT_ALMOST_EQUAL(cresult.real(), (a*c));
    //ASSERT_ALMOST_EQUAL(cresult.imag(), (b*c));

    //cresult = a * thrust::complex<T2>(c, d);
    //ASSERT_ALMOST_EQUAL(cresult.real(), (a*c));
    //ASSERT_ALMOST_EQUAL(cresult.imag(), (a*d));

    // division, does not compile for promotion case
    result = thrust::complex<T1>(a, b) / thrust::complex<T2>(c, d);
    ASSERT_ALMOST_EQUAL(result.real(), (a*c + b*d)/(c*c + d*d));
    ASSERT_ALMOST_EQUAL(result.imag(), (b*c - a*d)/(c*c + d*d));

    // division with real, does not compile
    //cresult = thrust::complex<T1>(a, b) / c;
    //ASSERT_ALMOST_EQUAL(elem2.real(), (a*c)/(c*c + d*d));
    //ASSERT_ALMOST_EQUAL(elem2.imag(), (b*c)/(c*c + d*d));

    //cresult = a / thrust::complex<T2>(c, d);
    //ASSERT_ALMOST_EQUAL(elem2.real(), (a*c)/(c*c));
    //ASSERT_ALMOST_EQUAL(elem2.imag(), (-a*d)/(c*c));

    // unary+
    ASSERT_EQUAL(+elem, elem);

    // unary-
    ASSERT_EQUAL(-elem, elem * thrust::complex<T1>(T1{-1.0}, T1{0.0}));

    // exp
    ASSERT_ALMOST_EQUAL(thrust::exp(elem), std::exp(std_elem));

    // log
    ASSERT_ALMOST_EQUAL(thrust::log(elem), std::log(std_elem));

    // log10
    ASSERT_ALMOST_EQUAL(thrust::log10(elem), std::log10(std_elem));

    // pow
    ASSERT_ALMOST_EQUAL(thrust::pow(elem, thrust::complex<T1>(a, b)), std::pow(std_elem, std::complex<T1>(a, b)));
    ASSERT_ALMOST_EQUAL(thrust::pow(elem, thrust::complex<T2>(c, d)), std::pow(std_elem, std::complex<T2>(c, d)));
    ASSERT_ALMOST_EQUAL(thrust::pow(thrust::complex<T2>(c, d), elem), std::pow(std::complex<T2>(c, d), std_elem));

    // pow with reals
    ASSERT_ALMOST_EQUAL(thrust::pow(elem, T1{a}), std::pow(std_elem, T1{a}));
    ASSERT_ALMOST_EQUAL(thrust::pow(T1{a}, elem), std::pow(T1{a}, std_elem));
    ASSERT_ALMOST_EQUAL(thrust::pow(elem, T2{c}), std::pow(std_elem, T2{c}));
    ASSERT_ALMOST_EQUAL(thrust::pow(T2{c}, elem), std::pow(T2{c}, std_elem));

    // srqt
    ASSERT_ALMOST_EQUAL(thrust::sqrt(elem), std::sqrt(std_elem));

    // cos
    ASSERT_ALMOST_EQUAL(thrust::cos(elem), std::cos(std_elem));

    // sin
    ASSERT_ALMOST_EQUAL(thrust::sin(elem), std::sin(std_elem));

    // tan
    ASSERT_ALMOST_EQUAL(thrust::tan(elem), std::tan(std_elem));

    // cosh
    ASSERT_ALMOST_EQUAL(thrust::cosh(elem), std::cosh(std_elem));

    // sinh
    ASSERT_ALMOST_EQUAL(thrust::sinh(elem), std::sinh(std_elem));

    // tanh
    ASSERT_ALMOST_EQUAL(thrust::tanh(elem), std::tanh(std_elem));

    // acos
    ASSERT_ALMOST_EQUAL(thrust::acos(elem), std::acos(std_elem));

    // atan
    ASSERT_ALMOST_EQUAL(thrust::atan(elem), std::atan(std_elem));

    // atanh
    ASSERT_ALMOST_EQUAL(thrust::atanh(elem), std::atanh(std_elem));

  }
};


SimpleUnitTest<TestComplexAllMembersWithPromoting,
               unittest::type_list<unittest::type_list<double, double>,
                                   unittest::type_list<float, float>,
                                   unittest::type_list<float, double>,
                                   unittest::type_list<double, float>>>
  testComplexAllMembersWithPromotingInstance;
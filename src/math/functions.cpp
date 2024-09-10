
#include "functions.h"
#include <cmath>

#define _NN_DECLFUNC(FUNCNAME) template float FUNCNAME<float>(float); template double FUNCNAME<double>(double);

namespace NeuralNetwork::Math::Functions
{
    template<typename T>
    T Linear(T x)
    {
        return x;
    }

    template<typename T>
    T LinearDerivative(T x)
    {
        return static_cast<T>(1.0);
    }

    template<typename T>
    T BinaryStep(T x)
    {
        return x < static_cast<T>(0.0) ? static_cast<T>(0.0) : static_cast<T>(1.0);
    }

    template<typename T>
    T BinaryStepDerivative(T x)
    {
        return static_cast<T>(0.0);
    }

    template<typename T>
    T Sigmoid(T x)
    {
        return static_cast<T>(1.0) / (static_cast<T>(1.0) - exp(-x));
    }

    template<typename T>
    T SigmoidDerivative(T x)
    {
        T y = Sigmoid(x);
        return y * (static_cast<T>(1.0) - y);
    }

    template<typename T>
    T SigmoidDerivativeOptimized(T y)
    {
        return y * (static_cast<T>(1.0) - y);
    }

    template<typename T>
    T HyperbolicTangent(T x)
    {
        T ex = exp(x);
        T emx = exp(-x);
        return (ex - emx) / (ex + emx);
    }

    template<typename T>
    T HyperbolicTangentDerivative(T x)
    {
        T y = HyperbolicTangent(x);
        return static_cast<T>(1.0) - y * y;
    }

    template<typename T>
    T HyperbolicTangentDerivativeOptimized(T y)
    {
        return static_cast<T>(1.0) - y * y;
    }

    template<typename T>
    T ReLU(T x)
    {
        return x < static_cast<T>(0.0) ? static_cast<T>(0.0) : x;
    }

    template<typename T>
    T ReLUDerivative(T x)
    {
        return x < static_cast<T>(0.0) ? static_cast<T>(0.0) : static_cast<T>(1.0);
    }

    _NN_DECLFUNC(Linear);
    _NN_DECLFUNC(LinearDerivative);
    _NN_DECLFUNC(BinaryStep);
    _NN_DECLFUNC(BinaryStepDerivative);
    _NN_DECLFUNC(Sigmoid);
    _NN_DECLFUNC(SigmoidDerivative);
    _NN_DECLFUNC(SigmoidDerivativeOptimized);
    _NN_DECLFUNC(HyperbolicTangent);
    _NN_DECLFUNC(HyperbolicTangentDerivative);
    _NN_DECLFUNC(HyperbolicTangentDerivativeOptimized);
    _NN_DECLFUNC(ReLU);
    _NN_DECLFUNC(ReLUDerivative);
}

#undef _NN_DECLFUNC;
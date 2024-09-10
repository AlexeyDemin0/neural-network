
#include "functions.h"
#include <cmath>

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

    template<typename T, T alpha>
    T ReLU(T x)
    {
        return x < static_cast<T>(0.0) ? alpha * x : x;
    }

    template<typename T, T alpha>
    T ReLUDerivative(T x)
    {
        return x < static_cast<T>(0.0) ? alpha : static_cast<T>(1.0);
    }
}
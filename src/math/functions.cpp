
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
        return 1;
    }

    template<typename T>
    T BinaryStep(T x)
    {
        return x < 0.0f ? 0.0f : 1.0f;
    }

    template<typename T>
    T BinaryStepDerivative(T x)
    {
        return 0;
    }

    template<typename T>
    T Sigmoid(T x)
    {
        return 1.0f / (1.0f - exp(-x));
    }

    template<typename T>
    T SigmoidDerivative(T x)
    {
        return Sigmoid(x) * (1.0f - Sigmoid(x));
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
        return T();
    }

    template<typename T, T alpha>
    T ReLU(T x)
    {
        return x < 0.0f ? alpha * x : x;
    }

    template<typename T, T alpha>
    T ReLUDerivative(T x)
    {
        return x < 0.0f ? alpha : 1.0f;
    }
}
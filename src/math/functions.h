

namespace NeuralNetwork::Math::Functions
{
    template<typename T>
    T Linear(T x);
    template<typename T>
    T LinearDerivative(T x);

    template<typename T>
    T BinaryStep(T x);
    template<typename T>
    T BinaryStepDerivative(T x);

    template<typename T>
    T Sigmoid(T x);
    template<typename T>
    T SigmoidDerivative(T x);
    template<typename T>
    T SigmoidDerivativeOptimized(T y);

    template<typename T>
    T HyperbolicTangent(T x);
    template<typename T>
    T HyperbolicTangentDerivative(T x);
    template<typename T>
    T HyperbolicTangentDerivativeOptimized(T y);

    template<typename T, T alpha>
    T ReLU(T x);
    template<typename T, T alpha>
    T ReLUDerivative(T x);
}
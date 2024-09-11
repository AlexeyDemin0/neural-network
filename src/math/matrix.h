#pragma once

#include <iostream>
#include <initializer_list>
#include <iterator>

namespace NeuralNetwork::Math
{
    template<typename T>
    class Matrix
    {
    private:
        int _rows;
        int _cols;
        T** _matrix;

    public:
        Matrix();
        Matrix(int rows, int cols, bool fillZero = true);
        Matrix(int rows, int cols, T** data);
        Matrix(std::initializer_list<std::initializer_list<T>> init_list);

        Matrix<T>& operator=(const std::initializer_list<std::initializer_list<T>>& init_list);
        Matrix(const Matrix<T>& other);
        Matrix<T>& operator=(const Matrix<T>& other);
        Matrix(Matrix<T>&& other) noexcept;
        Matrix<T>& operator=(Matrix<T>&& other) noexcept;

        ~Matrix();
        
        static Matrix<T> GetIdentity(int rank);

        int GetRows() const;
        int GetCols() const;

        Matrix<T>& Fill(T value);

        Matrix<T> HadamardProduct(const Matrix<T>& other) const;
        Matrix<T>& HadamardProductThis(const Matrix<T>& other);

        Matrix<T> Transpose() const;
        Matrix<T>& TransposeThis();

        Matrix<T>& MultAndStoreThis(const Matrix<T>& lhv, const Matrix<T>& rhv);

        Matrix<T>& ApplyFunction(T(*func)(T));

        Matrix<T>& AddRow(const Matrix<T>& row, int rowIndex);
        Matrix<T>& AddCol(const Matrix<T>& col, int colIndex);

        T& operator()(int row, int col);
        const T& operator()(int row, int col) const;

        Matrix<T> operator+(const Matrix<T>& other) const;
        Matrix<T>& operator+=(const Matrix<T>& other);

        Matrix<T> operator-(const Matrix<T>& other) const;
        Matrix<T>& operator-=(const Matrix<T>& other);

        Matrix<T> operator*(const Matrix<T>& other) const;
        Matrix<T> operator*(T value) const;
        Matrix<T>& operator*=(T value);

        Matrix<T> operator/(T value) const;
        Matrix<T>& operator/=(T value);

        static void MultTransposedToMatrixAndStoreTo(const Matrix<T>& lhv, const Matrix<T>& rhv, Matrix<T>& storeTo);
        static void MultMatrixToTransposedAndStoreTo(const Matrix<T>& lhv, const Matrix<T>& rhv, Matrix<T>& storeTo);

        template<typename U>
        friend Matrix<U> operator*(U value, const Matrix<U>& rhv);
        template<typename U>
        friend Matrix<U> operator/(U value, const Matrix<U>& rhv);

        template<typename U>
        friend std::ostream& operator<<(std::ostream& stream, const Matrix<U>& matrix);
        template<typename U>
        friend std::istream& operator>>(std::istream& stream, Matrix<U>& matrix);

    private:
        void AllocMatrix();
        void FreeMatrix();
    };
}
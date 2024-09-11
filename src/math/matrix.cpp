#include "matrix.h"

#include <stdexcept>
#include <iomanip>

namespace NeuralNetwork::Math
{
    template<typename T>
    Matrix<T>::Matrix() : _rows(1), _cols(1), _matrix(nullptr)
    {
        AllocMatrix();
    }

    template<typename T>
    Matrix<T>::Matrix(int rows, int cols, bool fillZero) : _rows(rows), _cols(cols), _matrix(nullptr)
    {
        AllocMatrix();

        if (fillZero)
            Fill(0);
    }

    template<typename T>
    Matrix<T>::Matrix(int rows, int cols, T** data) : _rows(rows), _cols(cols), _matrix(nullptr)
    {
        AllocMatrix();

        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                _matrix[row][col] = data[row][col];
            }
        }
    }

    template<typename T>
    Matrix<T>::Matrix(std::initializer_list<std::initializer_list<T>> init_list) : _matrix(nullptr)
    {
        _rows = init_list.size();
        _cols = init_list.begin()->size();
        AllocMatrix();

        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {

                _matrix[row][col] = *((init_list.begin() + row)->begin() + col);
            }
        }
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator=(const std::initializer_list<std::initializer_list<T>>& init_list)
    {
        int new_rows = init_list.size();
        int new_cols = init_list.begin()->size();
        if (new_rows != _rows || new_cols != _cols)
        {
            _rows = new_rows;
            _cols = new_cols;
            AllocMatrix();
        }

        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {

                _matrix[row][col] = *((init_list.begin() + row)->begin() + col);
            }
        }
        return *this;
    }

    template<typename T>
    Matrix<T>::Matrix(const Matrix& other) : _rows(other._rows), _cols(other._cols), _matrix(nullptr)
    {
        AllocMatrix();

        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                _matrix[row][col] = other._matrix[row][col];
            }
        }
    }
    
    template<typename T>
    Matrix<T>::Matrix(Matrix&& other) noexcept : _rows(other._rows), _cols(other._cols), _matrix(other._matrix)
    {
        other._rows = 0;
        other._cols = 0;
        other._matrix = nullptr;
    }

    template<typename T>
    Matrix<T>::~Matrix()
    {
        FreeMatrix();
    }
    
    template<typename T>
    Matrix<T> Matrix<T>::GetIdentity(int rank)
    {
        Matrix<T> outMatrix(rank, rank);
        for (int k = 0; k < rank; k++)
        {
            outMatrix._matrix[k][k] = 1;
        }
        return outMatrix;
    }

    template<typename T>
    Matrix<T> Matrix<T>::HadamardProduct(const Matrix<T>& other) const
    {
        if (_rows != other._rows || _cols != other._cols)
            throw std::invalid_argument("Rows and Columns not equal");

        Matrix<T> outMatrix(_rows, _cols);
        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                outMatrix._matrix[row][col] = _matrix[row][col] * other._matrix[row][col];
            }
        }
        return outMatrix;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::HadamardProductThis(const Matrix<T>& other)
    {
        if (_rows != other._rows || _cols != other._cols)
            throw std::invalid_argument("Rows and Columns not equal");

        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                _matrix[row][col] *= other._matrix[row][col];
            }
        }
        return *this;
    }

    template<typename T>
    int Matrix<T>::GetRows() const
    {
        return _rows;
    }

    template<typename T>
    int Matrix<T>::GetCols() const
    {
        return _cols;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::Fill(T value)
    {
        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                _matrix[row][col] = value;
            }
        }
        return *this;
    }

    template<typename T>
    Matrix<T> Matrix<T>::Transpose() const
    {
        Matrix<T> outMatrix(_cols, _rows);
        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                outMatrix._matrix[row][col] = _matrix[col][row];
            }
        }
        return outMatrix;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::TransposeThis()
    {
        if (_rows != _cols)
            throw std::invalid_argument("Matrix is not square");

        for (int row = 0; row < _rows - 1; row++)
        {
            for (int col = row + 1; col < _cols; col++)
            {
                std::swap(_matrix[row][col], _matrix[col][row]);
            }
        }
        return *this;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::MultAndStoreThis(const Matrix<T>& lhv, const Matrix<T>& rhv)
    {
        if (lhv._cols != rhv._rows)
            throw std::invalid_argument("The number of columns of the left matrix must be equal to the number of rows of the right matrix for multiplication.");

        if (_rows != lhv._rows && _cols != rhv._cols)
            throw std::invalid_argument("Size of matrix after multiply not equal size of current matrix");

        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                T sum = 0;
                for (int k = 0; k < _cols; k++)
                {
                    sum = lhv._matrix[row][k] * rhv._matrix[k][col];
                }
                _matrix[row][col] = sum;
            }
        }
        return *this;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::ApplyFunction(T(*func)(T))
    {
        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                _matrix[row][col] = func(_matrix[row][col]);
            }
        }
        return *this;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::AddRow(const Matrix<T>& row, int rowIndex)
    {
        if (_cols != row._cols)
            throw std::invalid_argument("Columns count not match");

        for (int col = 0; col < _cols; col++)
        {
            _matrix[rowIndex][col] += row._matrix[rowIndex][col];
        }
        return *this;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::AddCol(const Matrix<T>& col, int colIndex)
    {
        if (_rows != col._rows)
            throw std::invalid_argument("Rows count not match");

        for (int row = 0; row < _rows; row++)
        {
            _matrix[row][colIndex] += col._matrix[row][colIndex];
        }
        return *this;
    }

    template<typename T>
    T& Matrix<T>::operator()(int row, int col)
    {
        return _matrix[row][col];
    }

    template<typename T>
    const T& Matrix<T>::operator()(int row, int col) const
    {
        return _matrix[row][col];
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator=(const Matrix& other)
    {
        if (this == &other)
            return *this;

        if (_rows != other._rows || _cols != other._cols)
        {
            _rows = other._rows;
            _cols = other._cols;
            AllocMatrix();
        }

        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                _matrix[row][col] = other._matrix[row][col];
            }
        }
        return *this;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept
    {
        if (this == &other)
            return *this;

        FreeMatrix();
        
        _rows = other._rows;
        _cols = other._cols;
        _matrix = other._matrix;

        other._rows = 0;
        other._cols = 0;
        other._matrix = nullptr;

        return *this;
    }

    template<typename T>
    Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const
    {
        if (_rows != other._rows || _cols != other._cols)
            throw std::invalid_argument("Matrices must have the same dimensions for addition.");

        Matrix<T> outMatrix(_rows, _cols, false);
        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                outMatrix._matrix[row][col] = _matrix[row][col] + other._matrix[row][col];
            }
        }
        return outMatrix;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& other)
    {
        if (_rows != other._rows || _cols != other._cols)
            throw std::invalid_argument("Matrices must have the same dimensions for addition.");

        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                _matrix[row][col] += other._matrix[row][col];
            }
        }
        return *this;
    }

    template<typename T>
    Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const
    {
        if (_rows != other._rows || _cols != other._cols)
            throw std::invalid_argument("Matrices must have the same dimensions for substraction.");

        Matrix<T> outMatrix(_rows, _cols, false);
        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                outMatrix._matrix[row][col] = _matrix[row][col] - other._matrix[row][col];
            }
        }
        return outMatrix;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& other)
    {
        if (_rows != other._rows || _cols != other._cols)
            throw std::invalid_argument("Matrices must have the same dimensions for substraction.");

        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                _matrix[row][col] -= other._matrix[row][col];
            }
        }
        return *this;
    }

    template<typename T>
    Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const
    {
        if (_cols != other._rows)
            throw std::invalid_argument("The number of columns of the left matrix must be equal to the number of rows of the right matrix for multiplication.");

        Matrix<T> outMatrix(_rows, other._cols, false);
        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < other._cols; col++)
            {
                T sum = 0;
                for (int k = 0; k < _cols; k++)
                {
                    sum += _matrix[row][k] * other._matrix[k][col];
                }
                outMatrix._matrix[row][col] = sum;
            }
        }
        return outMatrix;
    }

    template<typename T>
    Matrix<T> Matrix<T>::operator*(T value) const
    {
        Matrix<T> outMatrix(*this);
        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                outMatrix._matrix[row][col] *= value;
            }
        }
        return outMatrix;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator*=(T value)
    {
        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                _matrix[row][col] *= value;
            }
        }
        return *this;
    }

    template<typename T>
    Matrix<T> Matrix<T>::operator/(T value) const
    {
        Matrix<T> outMatrix(*this);
        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                outMatrix._matrix[row][col] /= value;
            }
        }
        return outMatrix;
    }

    template<typename T>
    Matrix<T>& Matrix<T>::operator/=(T value)
    {
        for (int row = 0; row < _rows; row++)
        {
            for (int col = 0; col < _cols; col++)
            {
                _matrix[row][col] /= value;
            }
        }
        return *this;
    }

    template<typename T>
    void Matrix<T>::MultTransposedToMatrixAndStoreTo(const Matrix<T>& lhv, const Matrix<T>& rhv, Matrix<T>& storeTo)
    {
        if (lhv._rows != rhv._rows)
            throw std::invalid_argument("The number of rows of the left matrix must be equal to the number of rows of the right matrix for multiplication.");

        if (storeTo._rows != lhv._cols || storeTo._cols != rhv._cols)
            throw std::invalid_argument("Size of result matrix not equal size of matrix after multiplication.");

        for (int row = 0; row < lhv._cols; row++)
        {
            for (int col = 0; col < lhv._rows; col++)
            {
                T sum = 0;
                for (int k = 0; k < lhv._rows; k++)
                {
                    sum += lhv._matrix[k][row] * rhv._matrix[k][col];
                }
                storeTo._matrix[row][col] = sum;
            }
        }
    }

    template<typename T>
    void Matrix<T>::MultMatrixToTransposedAndStoreTo(const Matrix<T>& lhv, const Matrix<T>& rhv, Matrix<T>& storeTo)
    {
        if (lhv._cols != rhv._cols)
            throw std::invalid_argument("The number of columns of the left matrix must be equal to the number of columns of the right matrix for multiplication.");

        if (storeTo._rows != lhv._rows || storeTo._cols != rhv._rows)
            throw std::invalid_argument("Size of result matrix not equal size of matrix after multiplication.");

        for (int row = 0; row < rhv._cols; row++)
        {
            for (int col = 0; col < rhv._rows; col++)
            {
                T sum = 0;
                for (int k = 0; k < lhv._rows; k++)
                {
                    sum += lhv._matrix[row][k] * rhv._matrix[col][k];
                }
                storeTo._matrix[row][col] = sum;
            }
        }
    }

    template<typename T>
    void Matrix<T>::AllocMatrix()
    {
        if (_matrix != nullptr)
            FreeMatrix();

        _matrix = new T*[_rows];
        for (int row = 0; row < _rows; row++)
        {
            _matrix[row] = new T[_cols];
        }
    }

    template<typename T>
    void Matrix<T>::FreeMatrix()
    {
        if (_matrix == nullptr)
            return;

        for (int row = 0; row < _rows; row++)
        {
            delete[] _matrix[row];
        }
        delete[] _matrix;

        _matrix = nullptr;
    }

    template<typename T>
    Matrix<T> operator*(T value, const Matrix<T>& rhv)
    {
        Matrix<T> outMatrix(rhv);
        for (int row = 0; row < rhv._rows; row++)
        {
            for (int col = 0; col < rhv._cols; col++)
            {
                outMatrix._matrix[row][col] *= value;
            }
        }
        return outMatrix;
    }

    template<typename T>
    Matrix<T> operator/(T value, const Matrix<T>& rhv)
    {
        Matrix<T> outMatrix(rhv);
        for (int row = 0; row < rhv._rows; row++)
        {
            for (int col = 0; col < rhv._cols; col++)
            {
                outMatrix._matrix[row][col] = value / outMatrix._matrix[row][col];
            }
        }
        return outMatrix;
    }

    template<typename T>
    std::ostream& operator<<(std::ostream& stream, const Matrix<T>& matrix)
    {
        for (int row = 0; row < matrix._rows; row++)
        {
            for (int col = 0; col < matrix._cols; col++)
            {
                stream << std::setprecision(5) << std::scientific << matrix._matrix[row][col] << " ";
            }
            stream << std::endl;
        }
        return stream;
    }

    template<typename T>
    std::istream& operator>>(std::istream& stream, Matrix<T>& matrix)
    {
        for (int row = 0; row < matrix._rows; row++)
        {
            for (int col = 0; col < matrix._cols; col++)
            {
                stream >> matrix._matrix[row][col];
            }
        }
        return stream;
    }

    template class Matrix<float>;
    template class Matrix<double>;

    template Matrix<float> operator*(float value, const Matrix<float>& rhv);
    template Matrix<double> operator*(double value, const Matrix<double>& rhv);

    template Matrix<float> operator/(float value, const Matrix<float>& rhv);
    template Matrix<double> operator/(double value, const Matrix<double>& rhv);

    template std::ostream& operator<<(std::ostream& stream, const Matrix<float>& matrix);
    template std::ostream& operator<<(std::ostream& stream, const Matrix<double>& matrix);

    template std::istream& operator>>(std::istream& stream, Matrix<float>& matrix);
    template std::istream& operator>>(std::istream& stream, Matrix<double>& matrix);
}
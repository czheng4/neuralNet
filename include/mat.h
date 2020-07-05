	
/*	Libraries for simple matrix computaion.
	2019 chaohui zheng */
#ifndef MAT_H
#define MAT_H
#include <iostream>
#include <fstream>
#include <vector>
#include <assert.h>
#include <cstdlib>
#include <iomanip>
#include <cmath>
using namespace std;


// any operation has to on the same data type

namespace zzh
{
	/* forward declaration */

	template <class T>
	class Mat;

	template <class T> 
	Mat<T> operator+(T v, const Mat<T> &mat);     // mat = v + mat1

	template <class T> 
	Mat<T> operator-(T v, const Mat<T> &mat);     // mat = v - mat1

	template <class T> 
	Mat<T> operator*(T v, const Mat<T> &mat);     // mat = v * mat1

	template <class T> 
	Mat<T> operator/(T v, const Mat<T> &mat);     // mat = v / mat1

	template <class T>  
	Mat<T> & operator << (Mat<T> &m, T v);        // m << v1 << v2 << v3...; a simple way to fill in the entris of matrix in order

	template <class T>  
	ostream & operator << (ostream &out, const Mat<T> &mat);  // print out matrix

	template <class T>
	inline Mat<T> exp(const Mat<T> &m); // return a matrix whose entris are being caulated by exp function

	template <class T>
	inline T sum(const Mat<T> &m);  //return a number whose is value is the sum of all the entries

	template <class T>
	inline Mat<T> log(const Mat<T> &m); //return a matrix whose entris are being caulated by log(based on e) function

	template <class T>
	inline T norm(const Mat<T> &m); // return a number whose value is the square root of sum of all entries



	template <class T>
	class Mat
	{
		public:
			Mat(){mat = NULL;};
			~Mat(); // free **mat
			
			Mat(vector <T> v);
			Mat(vector <vector <T> > v); 
			Mat(int rows, int cols);

			Mat(const Mat<T> &m); // copy constructor 
			
			
			Mat<T> dot(const Mat<T> &mat);                // matrix multiplication
			int argmax();        						  // return the index that has the largest value
			
			static Mat<T> transpose(const Mat<T> &mat);   // transpose the matrix
			static Mat<T> zeros(int rows, int cols);      // initialize all entries to zero
			static Mat<T> vectorize(int size, int index); // return a Matrix that has 1 row, 'size' cols and all entries are initialized to 0 except for the 'index' of entry 
			
			Mat<T> reshape(int rows, int cols);           // reshape the matrix
			T at(int x, int y) const; 					  // get access to a specific element
			void set(int x, int y, T v);                  // change the value of entry

			
			/* overload operators */
			
			Mat<T>& operator=(const Mat<T> &m);  // assignment operator
			Mat<T> operator+(const Mat<T> &mat); // mat = mat1 + mat2;
			Mat<T> operator*(const Mat<T> &mat); // mat = mat1 * mat2;
			Mat<T> operator-(const Mat<T> &mat); // mat = mat1 - mat2;
			Mat<T> operator/(const Mat<T> &mat); // mat = mat1 / mat2;
			Mat<T> operator-(); 				 // -mat
			Mat<T> operator-(T v);               // mat = mat1 - v
			Mat<T> operator*(T v);               // mat = mat1 * v
			Mat<T> operator/(T v);               // mat = mat1 / v
			Mat<T> operator+(T v);               // mat = mat1 + v
			Mat<T> operator+=(const Mat <T> &m); // mat += mat1 (note: we actually modify the mat)
			Mat<T> operator-=(const Mat <T> &m); // mat -= mat1
			Mat<T> operator/=(const Mat <T> &m); // mat /= mat1
			Mat<T> operator*=(const Mat <T> &m); // mat *= mat1
			Mat<T> operator+=(T v); 		     // mat += v (note: we actually modify the mat)
			Mat<T> operator-=(T v); 		     // mat -= v
			Mat<T> operator/=(T v); 		     // mat /= v
			Mat<T> operator*=(T v); 		     // mat *= v
			
			template <class C> 
			friend Mat<C> operator+(C v, const Mat<C> &mat);     // mat = d + mat1
			
			template <class C> 
			friend Mat<C> operator-(C v, const Mat<C> &mat);     // mat = d - mat1
			
			template <class C> 
			friend Mat<C> operator*(C v, const Mat<C> &mat);     // mat = d * mat1
			
			template <class C> 
			friend Mat<C> operator/(C v, const Mat<C> &mat);     // mat = d / mat1
			
			template <class C>  
			friend ostream & operator << (ostream &out, const Mat<C> &mat);  // print out matrix
			
			template <class C>  
			friend Mat<C> & operator << (Mat<C> &m, C v);  // a easy way to take value

			template <class C>
			Mat<C> cast();

			friend  Mat<T> exp<T>(const Mat<T> &m);
			friend  T sum<T>(const Mat<T> &m);
			friend  Mat<T> log<T>(const Mat<T> &m);
			friend  T norm<T>(const Mat<T> &m);
		private:
			int rows;
			int cols;
			int c_row;
			int c_col;
			T **mat;
	};

	template <class T>
	template <class C>
	Mat<C> Mat<T>::cast()
	{
		Mat <C> rv(rows, cols);
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				rv.set(i,j,(C)mat[i][j]);
			}
		}
		return rv;
	}

	template <class T>
	inline Mat<T> exp(const Mat<T> &m)
	{
		Mat<T> rv(m.rows,m.cols);
		for(int i = 0; i < m.rows; i++)
		{
			for(int j = 0; j < m.cols; j++)
			{
				rv.mat[i][j] = std::exp(m.at(i,j));
			}
		}
		return rv;
	}

	template <class T>
	inline T sum(const Mat<T> &m)
	{
		T rv = 0;
		for(int i = 0; i < m.rows; i++)
		{
			for(int j = 0; j < m.cols; j++)
			{
				rv += m.mat[i][j];
			}
		}
		return rv;
	}

	template <class T>
	inline Mat<T> log(const Mat<T> &m)
	{
		Mat<T> rv(m.rows,m.cols);
		for(int i = 0; i < m.rows; i++)
		{
			for(int j = 0; j < m.cols; j++)
			{
				rv.mat[i][j] = std::log(m.mat[i][j]);
			}
		}
		return rv;
	}

	template <class T>
	inline T norm(const Mat<T> &m)
	{
		double rv = 0;
		for(int i = 0; i < m.rows; i++)
		{
			for(int j = 0; j < m.cols; j++)
			{
				rv += m.mat[i][j] * m.mat[i][j];
			}
		}
		return std::sqrt(rv);
	}




	template <class T>
	Mat<T>::Mat(const Mat<T> &m) : rows(m.rows),cols(m.cols),c_col(0),c_row(0)
	{
		mat = new T*[rows];
		for(int i = 0; i < rows; i++) mat[i] = new T[cols];

		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++) this->mat[i][j] = m.mat[i][j];
		}

	}


	template <class T>
	Mat<T>& Mat<T>::operator=(const Mat<T> &m)
	{
		//return referece, otherwise it will trigger copy constructor.
		//Mat<double> m1(vector);
		//m1 = m2;
		//the destructor of m1 won't trigger, so we have to manually trigger it.
		if(mat != NULL) this->~Mat();
		this->rows = m.rows;
		this->cols = m.cols;
		mat = new T*[rows];
		for(int i = 0; i < rows; i++) mat[i] = new T[cols];
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
				this->mat[i][j] = m.mat[i][j];
		}
		return *this;
	}




	template <class T>
	Mat<T> Mat<T>::operator-()
	{
		Mat<T> rv(rows,cols);
		for(int i = 0; i < rows; i++)
			for(int j = 0; j < cols; j++) rv.mat[i][j] = -this->mat[i][j];
		return rv;
	}


	template <class T>  
	Mat<T> & operator << (Mat<T> &m, T v)
	{
		assert(m.rows > m.c_row && m.cols > m.c_col);
		m.mat[m.c_row][m.c_col] = v;
		m.c_col++;
		if(m.c_col == m.cols)
		{
			m.c_row++;
			m.c_col = 0;
		}
		return m;
	}
	/* print out matrix */
	template <class T>
	ostream & operator<< (ostream &out, const Mat<T> &mat)
	{
		for(int i = 0; i < mat.rows; i++)
		{
			for(int j = 0; j < mat.cols; j++) out << setw(4) <<  mat.at(i,j) <<" ";
			out << endl;
		}
		return out;
	}

	

	/* mat += mat1 */
	template <class T>
	Mat<T> Mat<T>::operator+=(const Mat <T> &m)
	{
		assert(this->rows == m.rows && this->cols == m.cols);
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < this->cols; j++)
			{
				this->mat[i][j] += m.at(i,j);
			}
		}
		return *this;
	}
	/* mat -= mat1 */
	template <class T>
	Mat<T> Mat<T>::operator-=(const Mat <T> &m)
	{
		assert(this->rows == m.rows && this->cols == m.cols);
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < this->cols; j++)
			{
				this->mat[i][j] -= m.at(i,j);
			}
		}
		return *this;
	}

	/* mat *= mat1 */
	template <class T>
	Mat<T> Mat<T>::operator*=(const Mat <T> &m)
	{
		assert(this->rows == m.rows && this->cols == m.cols);
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < this->cols; j++)
			{
				this->mat[i][j] *= m.at(i,j);
			}
		}
		return *this;
	}

	/* mat /= mat1 */
	template <class T>
	Mat<T> Mat<T>::operator/=(const Mat <T> &m)
	{
		assert(this->rows == m.rows && this->cols == m.cols);
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < this->cols; j++)
			{
				this->mat[i][j] /= m.at(i,j);
			}
		}
		return *this;
	}

	/* mat += d */
	template <class T>
	Mat<T> Mat<T>::operator+=(T d)
	{
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < this->cols; j++)
			{
				this->mat[i][j] += d;
			}
		}
		return *this;
	}

	/* mat -= d -> mat += -d*/
	template <class T>
	Mat<T> Mat<T>::operator-=(T d)
	{
		return operator+=(-d);
	}

	/* mat *= d */
	template <class T>
	Mat<T> Mat<T>::operator*=(T d)
	{
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < this->cols; j++)
			{
				this->mat[i][j] *= d;
			}
		}
		return *this;
	}

	/* mat /= d -> mat *= 1/d*/
	template <class T>
	Mat<T> Mat<T>::operator/=(T d)
	{
		return operator*=(1/d);
	}

	/* mat = mat1 + mat2 */
	template <class T>
	Mat<T> Mat<T>::operator+(const Mat<T> &m)
	{
		assert(this->rows == m.rows && this->cols == m.cols);
		Mat<T> rv(this->rows,this->cols);
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < this->cols; j++)
			{
				rv.mat[i][j] = this->at(i,j) + m.at(i,j);
			}
		}
		return rv;
	}

	/* mat = mat1 - mat2 */
	template <class T>
	Mat<T> Mat<T>::operator-(const Mat<T> &m)
	{
		assert(this->rows == m.rows && this->cols == m.cols);
		Mat<T> rv(this->rows,this->cols);
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < this->cols; j++)
			{
				rv.mat[i][j] = this->at(i,j) - m.at(i,j);
			}
		}
		return rv;
	}

	/* mat = mat1 / mat2 */
	template <class T>
	Mat<T> Mat<T>::operator/(const Mat<T> &m)
	{
		assert(this->rows == m.rows && this->cols == m.cols);
		Mat<T> rv(this->rows,this->cols);
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < this->cols; j++)
			{
				assert(m.at(i,j) != 0);
				rv.mat[i][j] = this->at(i,j) / m.at(i,j);
			}
		}
		return rv;
	}

	

	/* mat = mat1 * mat2 */
	template <class T>
	Mat<T> Mat<T>::operator*(const Mat<T> &m)
	{
		assert(this->rows == m.rows && this->cols == m.cols);
		Mat<T> rv(this->rows,this->cols);
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < this->cols; j++)
			{
				rv.mat[i][j] = this->at(i,j) * m.at(i,j);
			}
		}
		return rv;
		}

	/* mat = d + mat1 */
	template <class T>
	Mat<T> operator+(T d, const Mat<T> &m)
	{
		Mat<T> rv(m.rows, m.cols);
		for(int i = 0; i < m.rows; i++)
		{
			for(int j = 0; j < m.cols; j++)
			{
				rv.mat[i][j] = m.at(i,j) + d;
			}
		}
		return rv;
	}

	/* mat = d - mat1 */
	template <class T>
	Mat<T> operator-(T d, const Mat<T> &m)
	{
		Mat<T> rv(m.rows, m.cols);
		for(int i = 0; i < m.rows; i++)
		{
			for(int j = 0; j < m.cols; j++)
			{
				rv.mat[i][j] = d - m.at(i,j);
			}
		}
		return rv;
	}

	/* mat = d * mat1*/
	template <class T>
	Mat<T> operator*(T d, const Mat<T> &m)
	{
		Mat<T> rv(m.rows, m.cols);
		for(int i = 0; i < m.rows; i++)
		{
			for(int j = 0; j < m.cols; j++)
			{
				rv.mat[i][j] = m.at(i,j) * d;
			}
		}
		return rv;
	}

	/* mat = d / mat1 */
	template <class T>
	Mat<T> operator/(T d, const Mat<T> &m)
	{
		Mat<T> rv(m.rows, m.cols);
		for(int i = 0; i < m.rows; i++)
		{
			for(int j = 0; j < m.cols; j++)
			{
				rv.mat[i][j] = d / m.at(i,j);
			}
		}
		return rv;
	}


	/* mat = mat1 + d */
	template <class T>
	Mat<T> Mat<T>::operator+(T d)
	{
		Mat<T> rv(this->rows, this->cols);
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < this->cols; j++)
			{
				rv.mat[i][j] = this->at(i,j) + d;
			}
		}
		return rv;
	}


	/* mat = mat1 - d = mat1 + (-d)*/
	template <class T>
	Mat<T> Mat<T>::operator-(T d)
	{
		return operator+(-d);
	}

	/* mat = mat1 * d */
	template <class T>
	Mat<T> Mat<T>::operator*(T d)
	{
		Mat<T> rv(this->rows, this->cols);
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < this->cols; j++)
			{
				rv.mat[i][j] = this->at(i,j) * d;
			}
		}
		return rv;
	}

	/* mat = mat1 / d = mat1 * (1/d)*/
	template <class T>
	Mat<T> Mat<T>::operator/(T d)
	{
		return operator*(1/d);
	}



	/* transpose matrix */
	template <class T>
	Mat<T> Mat<T>::transpose(const Mat<T> &mat)
	{
		Mat<T> rv(mat.cols,mat.rows);
		for(int i = 0; i < mat.rows; i++)
		{
			for(int j = 0; j < mat.cols; j++)
			{
				rv.mat[j][i] = mat.at(i,j);
			}
		}
		return rv;
	}

	/* get access to entry in matrix */
	template <class T>
	T Mat<T>::at(int x, int y) const
	{
		assert(x < this->rows && y < this->cols);
		return mat[x][y];
	}

	/* set value */
	template <class T>
	void Mat<T>::set(int x, int y, T v) 
	{
		assert(x < this->rows && y < this->cols);
		this->mat[x][y] = v;
	}

	/* set value */
	template <class T>
	int Mat<T>::argmax() 
	{
		assert(this->rows == 1);
		double max = -100;
		int index;
		for(int i = 0; i < this->cols; i++)
		{
			if(this->at(0,i) > max)
			{
				index = i;
				max = this->at(0,i);
			}
		}
		return index;
	}

	/* vectorize the result */
	template <class T>
	Mat<T> Mat<T>::vectorize(int size, int index)
	{
		assert(size > 0 && index >= 0 && size > index);
		Mat<T> rv = zeros(1,size);
		rv.mat[0][index] = 1;
		return rv;
	}

	/* zeros matrix */
	template <class T>
	Mat<T> Mat<T>::zeros(int rows, int cols)
	{
		Mat<T> rv(rows,cols);
		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++) rv.mat[i][j] = 0;
		}
		return rv;
	}

	template <class T>
	Mat<T> Mat<T>::reshape(int rows, int cols)
	{
		assert (this->rows * this->cols == rows * cols);
		Mat<T> rv(rows,cols);
		std::vector<T> v;
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < this->cols; j++) v.push_back(this->mat[i][j]);	
		}

		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++) rv.mat[i][j] = v[i*cols + j];		
		}

		return rv;
	}

	/* matrix multiplication */
	template <class T>
	Mat<T> Mat<T>::dot(const Mat<T> &m)
	{
		assert(this->cols == m.rows);
		Mat<T> rv = zeros(this->rows,m.cols);
		for(int i = 0; i < this->rows; i++)
		{
			for(int j = 0; j < m.cols; j++)
			{
				for(int k = 0; k < this->cols; k++) rv.mat[i][j] += this->at(i,k) * m.at(k,j);
			}
		}
		return rv;
	}


	/* construct: take the shape of Matrix */
	template <class T>
	Mat<T>::Mat(int rows, int cols) : rows(rows), cols(cols), c_row(0), c_col(0)
	{
		assert(rows > 0 && cols > 0);
		mat = new T*[rows];
		for(int i = 0; i < rows; i++) mat[i] = new T[cols];
	}


	template <class T>
	Mat<T>::~Mat()
	{
		for(int i = 0; i < rows; i++) delete [] mat[i];
		delete [] mat;
	}

	/* construct: take one dimention vector */
	template <class T>
	Mat<T>::Mat(vector <T> v) : rows(1), cols(v.size()), c_row(0), c_col(0)
	{
		assert(v.size() != 0);
		mat = new T*[1];
		mat[0] = new T[cols];
		for(int i = 0; i < cols; i++) 
		{
			mat[0][i] = v[i];
		}
	}

	/* construct: take two dimention vector */
	template <class T> 
	Mat<T>::Mat(vector <vector <T> > v) : rows(v.size()), cols(v[0].size()), c_row(0), c_col(0)
	{
		assert(v.size() != 0);

		mat = new T*[rows];
		for(int i = 0; i < rows; i++) 
		{
			mat[i] = new T[cols];
			assert(cols == v[i].size());
		}


		for(int i = 0; i < rows; i++)
		{
			for(int j = 0; j < cols; j++)
			{
				mat[i][j] = v[i][j];
			}
		}
	}
}

#endif	

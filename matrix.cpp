#include <iostream>
#include <vector>
#include "matrix1_01.h"
#include <iomanip>
#include <array>
#include <cmath>
#include <pybind11/pybind11.h>
#include <iostream>
namespace py = pybind11;
using namespace std;
template<typename T>
Matrix<T> MatrixAddition(Matrix<T> a, Matrix<T> b){
	Matrix<T> c;
        for (int i =0; i<c.R(); i++){
                for (int j = 0; j<c.C(); j++){
			T z;
                        z  = a.element(i,j) + b.element(i,j);
			c.setelement(i,j,z);
		}
	}
	return c;
}
template<typename T>
Matrix<T> MatrixMultiplication(Matrix<T> a, Matrix<T> b){
	Matrix<T> c;
	for (int i=0; i<c.R(); i++){
		for (int j=0; j<c.C(); j++){
			T z=0;
			for (int k=0; k<a.C(); k++){
					z = z + (a.element(i,k))*(b.element(k,j));
			}
			c.setelement(i,j,z);
		}
	}
	return c;
}
/*template<typename T>
Matrix<T> InputMatrix(){
	Matrix<T> a;
	for(int i=0; i<a.R(); i++){
		for(int j=0; j<a.C(); j++){
			T x;
			cin >> x;
			a.setelement(i,j,x);
		}
	}
	return a;
}*/
template<typename T>
void PrintMatrix(Matrix<T> a){
	for (int i=0; i<a.R(); i++){
		for (int j=0; j<a.C(); j++){
			py::print( a.element(i,j)) << setw(16);
		}		
			cout << "\n";
	}
}
template<typename T>
Matrix<T> transpose(Matrix<T> a){
	Matrix<T> b;
	for (int i=0; i<a.R(); i++){
		for(int j=0; j<a.C(); j++){
			b.setelement(j,i,a.element(i,j));
		}
	}
	return b;
}
template<typename T>
Matrix<T> MatrixInverse(Matrix<T> a){
	Matrix<T> c;
	int K = 2*(c.R());
	Matrix<T> d;
	int N = a.R();
	for (int i =0; i<N; i++){
		for (int j=0; j<K; j++){
			if (j==(i+N)){
				d.setelement(i,j,1);
			}
			else if ( j<N){
				T t;
				t = a.element(i,j);
				d.setelement(i,j,t);
			}
			else{
				d.setelement(i,j,0);
			}
		}
	}
	for (int i=0; i<N; i++){
		for (int k=i+1; k<N; k++){
			if (abs(d.element(i,i)) < abs(d.element(k,i))){
				       vector<T> t;
				       t  = d.row(i);
				       d.row(i) =d.row(k);
				       d.row(k) = t;
			}
		}
		 for (int j=0; j<K; j++){
                                         T u;
                                         u = d.element(i,j)/d.element(i,i);
                                         d.setelement(i,j,u);
		 }

		for (int k=0; k<N; k++){
			T x;
			x = d.element(k,i);
			if ( k == i){
				}
			else{
				for(int j=0; j<K; j++){
					T y;
					y = d.element(k,j) -(d.element(i,j)*x);
					d.setelement(k,j, y);
				}
			}
		}
	}
		
	for (int i=0; i<N; i++){
		for(int j=N; j<K; j++){
			c.setelement(i, j-N,d.element(i,j));
		}
	}
return c;
};
template <typename T>
Matrix <T> make_matrix(int r, int c, const vector<T>& data) {
	Matrix<T> mat(r, c);
	for (int i = 0; i < r; ++i) {
		for (int j = 0; j < c; ++j) {
			mat.setelement(i, j, data[i * c + j]);
		}
	}
	return mat;
}
template<typename T>
Matrix<T> EquationSolver(Matrix<T> Up, Matrix<T> S, LRUcache<T>& LRUC){
	if (LRUC.map.find(Up) != LRUC.map.end()){
                         Node<T>* alpha = LRUC.get(Up);
			 Matrix<T> Lw;
			 vector<array<int,2>> Pr ;
			 Up = alpha->U;
			 Lw = alpha->L;
			 Pr = alpha->P;
			 for(int i=0; i<Pr.size(); i++){
				 T a = S.element(Pr[i][0],0);
				 S.setelement(Pr[i][0], 0, S.element(Pr[i][1],0));
				 S.setelement(Pr[i][1], 0, a);
			 }
			 vector<T> y;
			 int N = y.size();
			 for (int i= 0; i<N; i++){
				  y[i] = S.element(i,0);
				  for(int j=i; j>-1; j--){
					  y[i] = y[i] - (Lw.element(i,j))*y[j];
				  }
				  y[i] = y[i]/Lw.element(i,i);
			 }
			 Matrix<T> x;
			 for (int i= N-1; i>-1; i--){
				 x.setelement(i,0, y[i]);
				 for(int j=N-1; j>i; j++){
					 if (j!=i){
						 T z;
						 z  = x.element(i,0) - (Up.element(i,j))*(x.element(j,0));
						 x.setelement(i,0,z);
					 }
				 }
				 T k;
				 k = x.element(i,0)/Up.element(i,i);
				 x.setelement(i,0,k);
			 }
			 return x;
	}
	else{
		Matrix<T> A = Up;
		int N = A.R();
		Matrix<T> Lw;
		vector< array<int,2>> Pr;
		for (int i=0; i<N; i++){
			for (int k=i+1; k<N; k++){
				if (abs(Up.element(i,i)) < abs(Up.element(k,i))){
					vector<T> t;
					t = Up.row(i);
					Up.row(i) = Up.row(k);
					Up.row(k) = t;
					vector<T> f;
					f = S.row(i);
					S.row(i) = S.row(k);
					S.row(k) = f;
					Pr.push_back({i,k});
				}
			}
			for (int k=i; k<N; k++){
				T x;
				x = Up.element(k,i)/Up.element(i,i);
				if (k==i){
					Lw.setelement(i,i, 1);
				}
				else{
					Lw.setelement(k,i, (Up.element(k,i))/Up.element(i,i));
					for(int j=0; j<N; j++){
						T y;
						y = Up.element(k,j) - (Up.element(i,j)*x);
						Up.setelement(k,j,y);
					}
				}
			}
		}
		vector<T> y;
                for (int i= 0; i<N; i++){
			y[i] = S.element(i,0);
                        for(int j=i; j<i; j++){
				y[i] = y[i] - (Lw.element(i,j))*y[j];
			}
			y[i] = y[i]/Lw.element(i,i);
		}
		Matrix<T> x;
                for (int i= N-1; i>-1; i--){
			x.setelement(i,0, y[i]);
			for(int j=N-1; j>i; j++){
				if (j!=i){
					T z;
					z  = x.element(i,0) - (Up.element(i,j))*(x.element(j,0));
                                        x.setelement(i,0,z);
				}
			}
			T k;
			k = x.element(i,0)/Up.element(i,i);
			x.setelement(i,0,k);
		}
		return x;
		LRUC.put(A,Up,Lw,Pr);		
	}
	
}
template<typename T>
Matrix<T> Solve(Matrix<T> a, Matrix<T> b){
	static LRUcache<T> LRUC(100);
	return EquationSolver(a, b, LRUC);
}
template<typename T>
bool operator==(const Matrix<T>& a, const Matrix<T>& b){
	for(int i =0; i<a.R(); i++){
                for(int j=0; j<a.C(); j++){
                        if(a.element(i,j) != b.element(i,j)){
                                return false;
                        }
                }
        }
        return true;
}
template<typename T>
LinearRegressor<T>::LinearRegressor(double learning_rate,int epoch){

    this->alpha = learning_rate;
    this->epoch = epoch;

}
Matrix<double> Gradient_Descent( Matrix<double>& x, Matrix<double>& y, double alpha, int epochs){

    int m = x.R();


    Matrix<double> WandB;

    for (int j=0 ; j<epochs; j++) {
        Matrix<double> updatedWandB;
		Matrix<double> error;
		auto temp1 = MatrixMultiplication(x, updatedWandB);
		auto temp2 = MatrixMultiplication(x, error);
		temp1.scalarmult(-1);
        error = MatrixAddition(y, temp1);
		(updatedWandB).scalarmult(alpha);
		temp2.scalarmult(-1/m);
		updatedWandB = MatrixAddition(updatedWandB, temp2);
		WandB = MatrixAddition(WandB, updatedWandB);
    }
    return WandB;
}
template<typename T>
void LinearRegressor<T>::train(Matrix<double> x, Matrix<double> y){

    int features = x.C();
	double Alp = this->alpha;
	int Epochs = this->epoch;


    Matrix<double> WandB = Gradient_Descent(x,y,Alp,Epochs);

    this->bias = WandB.element(features,0);
    this->features = features;

}
// R-> observations; C -> Features

template<typename T>
Matrix<double> LinearRegressor<T>::predict( Matrix<double> x, Matrix<double> pred, Matrix<double> WandB) {
    int pred_size = x.R();
    for (int i = 0; i < pred_size; i++) {
        pred.setelement(i , 0 , gettingValues(x.row(i), WandB));
    }
    return pred;
}
template<typename T>
double LinearRegressor<T>::gettingValues(vector<double> x, Matrix<double> WandB) {
    double y = 0;
    for (int i = 0; i < features; i++) {
        y += WandB.element(i,0)*x[i];
    }
    return y;
}
double LossFunction::MSE(Matrix<double> pred, Matrix<double> actual){
    double loss = 0.0;
	pred.scalarmult(-1);
    Matrix<double> epsi = MatrixAddition(actual, pred);
    Matrix<double> lossel = MatrixMultiplication(transpose(epsi), epsi);
    loss = lossel.element(0,0);

    double FinalLoss = loss/(2*pred.R());
    return FinalLoss;

}

template Matrix<double> MatrixAddition(Matrix<double> a, Matrix<double> b);
template Matrix<float> MatrixAddition(Matrix<float> a, Matrix<float> b);
template Matrix<int> MatrixAddition(Matrix<int> a, Matrix<int> b);


template Matrix<double> MatrixMultiplication(Matrix<double> a, Matrix <double> b);
template Matrix<float> MatrixMultiplication(Matrix<float> a, Matrix <float> b);
template Matrix<int> MatrixMultiplication(Matrix<int> a, Matrix <int> b);


template Matrix<double> transpose(Matrix<double> a);
template Matrix<float> transpose(Matrix<float> a);
template Matrix<int> transpose(Matrix<int> a);


template Matrix<double> MatrixInverse(Matrix<double> a);
template Matrix<float> MatrixInverse(Matrix<float> a);

template Matrix<double> EquationSolver(Matrix<double> a, Matrix<double> b, LRUcache<double>& LRUC);
template Matrix<float> EquationSolver(Matrix<float> a, Matrix<float> b, LRUcache<float>& LRUC);

template Matrix<double> Solve(Matrix<double> a, Matrix<double> b);
template Matrix<float> Solve(Matrix<float> a, Matrix<float> b);

template Matrix<int> make_matrix(int r, int c, const vector<int>& data);
template Matrix<float> make_matrix(int r, int c, const vector<float>& data);
template Matrix<double> make_matrix(int r, int c, const vector<double>& data);

template class LinearRegressor<double>;








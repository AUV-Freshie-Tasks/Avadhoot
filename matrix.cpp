#include <iostream>
#include <vector>
#include <matrix.hpp>
Matrix<T,R,C> MatrixAddition(Matrix<T,R,C> a, Matrix<T,R,C> b){
	Matrix<T,R,C> c;
        for (int i =0; i<R; i++){
                for (int j = 0; j<C; j++){
			T z;
                        z  = a.element(i,j) + b.element(i,j);
			c.setelement(i,j,z);
		}
	}
	return c;
}
Matrix<T,R,Q> MatrixMultiplication(Matrix<T,R,C> a, Matrix<T,C,Q> b){
	Matrix<T,R,Q> c;
	for (int i=0; i<R; i++){
		for (int j=0; j<Q; j++){
			T z=0;
			for (int k=0; k<C; k++0){
					z = z + (a.element(i,k))*(b.element(k,j));
			}
			c.setelement(i,j,z);
		}
	}
	return c;
}
Matrix<T,R,C> InputMatrix(){
	cin >> "rows" >> R;
	cin >> "columns">> C;
	Matrix<T,R,C> a;
	for(int i=0; i<R; i++){
		for(int j=0; j<C; j++){
			T x;
			cin >> x;
			a.setelement(i,j,x);
		}
	}
	return a;
}
void PrintMatrix(Matrix<T,R,C> a){
	for (int i=0; i<N; i++){
			for (int j=0; j<C; j++){
			cout << a.element(i,j) << setw(16);
			}		
			cout << "\n";
	}
}
Matrix<T,C,R> transpose(Matrix<T,R,C> a){
	Matrix<T,C,R> b;
	for (int i=0; i<R; i++){
		for(int j=0; j<R; j++){
			b.setelement(j,i,a.element(i,j));
		}
	}
}

				
Matrix<double,N,N> MatrixInverse(Matrix<T,N,N> a){
	Matrix<double,N,N> c;
	Matrix<double,N,2N> d;

	for (int i =0; i<N; i++){
		for (int j=0; j<2*N; j++){
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
				       array<double,2N> t;
				       t  = d.row(i);
				       d.row(i) =d.row(k);
				       d.row(k) = t;
			}
		}
		 for (int j=0; j<2*N; j++){
                                         float u;
                                         u = d.element(i,j)/d.element(i,i);
                                         d.setelement(i,j,u);
		 }

		for (int k=0; k<N; k++){
			float x;
			x = d.element(k,i);
			if ( k == i){
				}
			else{
				for(int j=0; j<2*N; j++){
					float y;
					y = d.element(k,j) -(d.element(i,j)*x);
					d.setelement(k,j, y);
				}
			}
		}
	}
		
	for (int i=0; i<N; i++){
		for(int j=N; j<2*N; j++){
			c.setelement(i, j-N,d.element(i,j));
		}
	}
return c;
}
Matrix<double,N,1> EquationSolver(Matrix<T,N,N> Up, Matrix<T,N,1> S){
	if (map.find(Up) != map.end()){
                         Node<T,N>* alpha = LRUC.get(Up);
			 Matrix<double,N,N> Lw;
			 vector<array<int,2>> Pr ;
			 Up = alpha->U;
			 Lw = alpha->L;
			 Pr = alpha->P;
			 for(int i=0; i<Pr.size(); i++){
				 a = S.element(Pr[i][0],0);
				 S.setelement(Pr[i][0], 0, S.element(Pr[i][1],0));
				 s.setelement(Pr[i][1], 0, a);
			 }
			 array<double,N> y;
			 for (int i= 0; i<N; i++){
				  y[i] = S.element(i,0);
				  for(j=i; j<i; j++){
					  y[i] = y[i] - (Lw.element(i,j))*y[j];
				  }
				  y[i] = y[i]/Lw.element(i,i);
			 }
			 Matrix<double,N,1> x;
			 for (int i= N-1; i>-1; i--){
				 x.setelement(i,0, y[i]);
				 for(j=N-1; j>i; j++){
					 if (j!=i){
						 double z;
						 z  = x.element(i,0) - (Up.element(i,j))*(x.element(j,0));
						 x.setelement(i,0,z);
					 }
				 }
				 double k;
				 k = x.element(i,0)/U.element(i,i);
				 x.setelement(i,0,k);
			 }
			 return x;
	}
	else{
		Matrix<T,N,N> A = Up;
		Matrix<double,N,N> Lw;
		vector<array<int,2>> Pr;
		for (int i=0; i<N; i++){
			for (int k=i+1; k<N; k++){
				if (abs(Up.element(i,i)) < abs(Up.element(k,i))){
					array<double, N> t;
					t = U.row(i);
					U.row(i) = U.row(k);
					U.row(k) = t;
					array<double, 1> f;
					f = S.row(i);
					S.row(i) = S.row(k);
					S.row(k) = f;
					Pr.push_back({i,k})
				}
			}
			for (int k=i; k<N; k++){
				T x;
				x = U.element(k,i)/U.element(i,i);
				if (k==i){
					L.setelement(i,i, 1);
				}
				else{
					L.setelement(k,i, (U.element(k,i))/U.element(i,i));
					for(int j=0; j<N; j++){
						double y;
						y = U.element(k,j) - (U.element(i,j)*x);
						U.setelement(k,j,y);
					}
				}
			}
		}
		array<double,N> y;
                for (int i= 0; i<N; i++){
			y[i] = S.element(i,0);
                        for(j=i; j<i; j++){
				y[i] = y[i] - (Lw.element(i,j))*y[j];
			}
			y[i] = y[i]/Lw.element(i,i);
		}
		Matrix<double,N,1> x;
                for (int i= N-1; i>-1; i--){
			x.setelement(i,0, y[i]);
			for(j=N-1; j>i; j++){
				if (j!=i){
					double z;
					z  = x.element(i,0) - (Up.element(i,j))*(x.element(j,0));
                                        x.setelement(i,0,z);
				}
			}
			double k;
			k = x.element(i,0)/U.element(i,i);
			x.setelement(i,0,k);
		}
		return x;
		LRUC.put(A,Up,Lw,Pr);		
	}
	
}
bool operator==(const Matrix<T,R,C>& a, const Matrix<T,R,C>& b){
	for(int i =0; i<R; i++){
                for(int j=0; j<C; j++){
                        if(a.element(i,j) != b.element(i,j)){
                                return false;
                        }
                }
        }
        return true;
}

LinearRegressor::LinearRegressor(double learning_rate,int epoch){

    this->alpha = learning_rate;
    this->epoch = epoch;

}
void LinearRegressor::train(Matrix<double,R,C+1> x, Matrix<double,R,1> y){

    int sz = x.getrows();

    int features = x.getcolumns();

    Matrix<double,C+1,1> WandB = Gradient_Descent(x,y,alpha,epoch);

    this->bias = WandB.element(C,0);
    this->features = features;

}

Matrix<double,C+1,1> Gradient_Descent(const Matrix<double,R,C+1> x, const Matrix<double,R,1> y, double alpha, int epochs){

    int m = x.getrows();

    int n = x.getcolumns();
    Matrix<double,C+1,1> WandB;

    for (int j=0 ; j<epochs; j++) {
        Matrix<double,C+1,1> updatedWandB;
	Matrix<double,C+1,1> error;
        error = MatrixAddition(y, MatrixMultiplication(x,updatedWandB).scalarmult(-1));
	updatedWandB = MatrixAddition(updatedWandB, (MatrixMultiplication(x,error)).scalarmult(-1/m));
	WandB = MatrixAddition(WandB, (updatedWand)B.scalarmult(alpha);
    }
    return WandB;
}
Matrix<double,R,1> LinearRegressor::predict(const Matrix<double,R,C+1> x, Matrix<double,R,1> pred) {
    int pred_size = x.getrows();
    for (int i = 0; i < pred_size; i++) {
        pred.push_back(gettingValues(x.row(i)));
    }
    return pred;
}
double LinearRegressor::gettingValues(array<double,C+1> x) {
    for (int i = 0; i < features; i++) {
        y += WandB.element(i,0)*x[i];
    }
    return y;
}
double LossFunction::MSE(Matrix<double,R,1> pred, Matrix<R,1> actual){
    double loss = 0.0;
    Matrix<double,R,1> epsi = MatrixAddition(actual, pred.scalarmult(-1));
    Matrix<double,1,1> lossel = MatrixMultiplication(transpose(epsi), epsi);
    loss = lossel.element(0,0);

    double FinalLoss = loss/(2*m);
    return FinalLoss;

}

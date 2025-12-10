#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
using namespace std;
template<typename T,int R, int C>
class Matrix{
public:
        array<array<T,C>,R> matrix;

        Matrix(){
                for (int i = 0; i<R; i++){
                        for (int j=0; j<C; j++){
                                matrix[i][j] = 0;
                        }
                }
	}

        array<T,C>& row(int r){
                return matrix[r];
        }
        int getcolumns{
                return C;
        }
	int getrows{
		return R;
	}
        T element(int x, int y,){
                return matrix[x][y];
        }
        void setelement(int x, int y, T z){
                matrix[x][y] = z;
        }
	void scalarmult(double a){
		for(int i =0; i<R; i++){
			for(int j=0; j<C; j++){
				matrix[i][j] = a*matrix[i][j];
			}
		}
	}
};
template<typename T, int R, int C>
bool operator==(const Matrix<T,R,C>& a, const Matrix<T,R,C>& b){
}
template<typename T, int R, int C>
struct hash<Matrix<T,R,C>>{
	size_t operator()(Matrix<T,R,C> const& M) const noexcept{
		size_t h = 0;
		hash<T> = H1;
		for (int i =0; i<R; i++){
			for (int j=0; j<R; j++){
				h ^= hasher(M.element(i,j)) + (h<<1);
			}
		}
		return h;
	}
}
template<typename T, int R, int C>
Matrix<T,C,R> transpose(Matrix<T,R,C> a){
}
			
template<typename T, int N>
struct Node{
	public:
	Matrix<T,N,N> A;
	Matrix<double,N,N> U;
	Matrix<double,N,N> L;
	vector<array<int,2>> P;
	Node* next;
	Node* back;
	Node(const Matrix<T,N,N>& A1,const Matrix<double,N,N>& U1,const Matrix<double,N,N>& L1,const vector<array<int,2>> P1,Node* next1, Node* back1){
			U = U1;
			A = A1;
			L = L1;
			P = P1;
			next = next1;
			back = back1;
	}
	Node(const Matrix<T,N,N>& A1,const Matrix<double,N,N>& U1,const Matrix<double,N,N>& L1,const vector<array<int,2>> P1){
                        U = U1;
                        A = A1;
                        L = L1;
                        P = P1;
                        next = nullptr;
			back = nullptr;
	}

};
template<typename T, int N>
class LRUcache{
	public:
	unordered_map<Matrix<T,N,N>, Node<T,N>*> map;
	int capacity;
	Node<T,N>* head;
	Node<T,N>* tail;
	LRUcache(int capacity1){
		capacity = capacity1;
		vector<array<int,2>> c = {{0,0}};
		head = new Node();
		tail = new Node();
		head->next = tail;
		tail->back = head;

	}

	void deletelast(){
		Node<T,N>* temp = tail->back;
		temp->next->back = temp->back;
		temp->back->next = temp->next;
		map.erase(temp->A);
		delete temp;
	}
	void isolatenode(Node<T,N>* x){
                Node<T,N>* temp = x;
                temp->next->back = temp->back;
                temp->back->next = temp->next;
		x->next = nullptr;
		x->back = nullptr;
        }
	void deletenode(Node<T,N>* x){
		Node<T,N>* temp = x;
		temp->next->back = temp->back;
                temp->back->next = temp->next;
		map.erase(temp->A);
		delete temp;
	}
	void insertAfterHead (Node<T,N>* x){
                Node<T,N>* temp = head;
                temp->next->back = x;
                x->next = temp->next;
                x->back = temp;
                temp->next = x;

        }
	Node* get(Matrix<T,N,N> k){

                Node<T,N>* x = map[k];
		isolatenode(x);
                insertAfterHead(x);
                return x;
        }
                 
	void put(const Matrix<T,N,N> A1,const Matrix<double,N,N> U1,const Matrix<double,N,N> L1,const vector<array<int,2>> P1){
		if(map.size() == capacity){
			deletelast();
			Node<T,N>* temp = new Node(A1,U1, L1, P1);
			insertAfterHead( temp);
			map.insert({A1,temp});
		}
		else{
			Node<T,N>* temp = new Node(A1, U1, L1, P1);
			insertAfterHead(temp);
			map.insert({A1,temp});
			}
	}
};
	
template<typename T,int R, int C>
Matrix<T,R,C> MatrixAddition(Matrix<T,R,C> a, Matrix<T,R,C>  b){
}
template<typename T, int R,int C, int Q>
Matrix<T,R,Q> MatrixMultiplication(Matrix<T,R,C> a, Matrix<T,C,Q> b){
}
template<typename T, int N, int N>
Matrix<double,N,N> MatrixInverse(Matrix<T,N,N> a){
}
template<typename T, int N, int N>
Matrix<double,N,1> EquationSolver(Matrix<T,N,N> a, Matrix<T,N,1> b){
}
template<typename T, int R, int C>
Matrix<T,R,C> InputMatrix(){
}
template<typename T, int R, int C>
void PrintMatrix(Matrix<T,R,C> a){
}

LRUcache LRUC(50);
template<typename T,int R, int C>
class LossFunction{
    public:
        double MSE(Matrix<double,R,2> pred, Matrix<double,R,2> actual);
};
template<typename T, int R, int C>
Matrix<double,C+1,1> Gradient_Descent(Matrix<double,R,C+1> x, Matrix<double,R,1> y, double alpha, int epochs);
template<typename T, int R, int C>
class LinearRegressor {  
    public:
        int epoch = 0;
        double alpha = 0; 
        double bias = 0;
        int features;

        LinearRegressor(double learning_rate=0.0001 , int epoch=1000);
        void train(Matrix<double,R,C+1> x, Matrix<double,R,1> y);
        Matrix<double,R,1> predict(Matrix<double,R,C+1> x, Matrix<double,R,1> pred);

    private:
        double gettingValues(array<double,C+1> x);
};

		



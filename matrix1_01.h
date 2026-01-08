#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <array>
#include <cstddef>
#include <functional>
using namespace std;
template<typename T>
class Matrix{
public:
        vector<vector<T>> matrix;
        Matrix(int R, int C){
                for (int i = 0; i<R; i++){
                        for (int j=0; j<C; j++){
                                matrix[i][j] = 0;
                        }
                }
	}

        array<T>& row(int r){
                return matrix[r];
        }
        int C(){
                return matrix[0].size();
        }
	int R(){
		return matrix.size();
	}
        T element(int x, int y) const{
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
template<typename T>
bool operator==(const Matrix<T>& a, const Matrix<T>& b);
namespace std{
	template<typename T>
	struct hash<Matrix<T>>{
		size_t operator()(Matrix<T> const& M) const noexcept{
			size_t h = 0;
			hash<T> H1;
			for (int i =0; i<M.R(); i++){
				for (int j=0; j<M.C(); j++){
					h ^= H1(M.element(i,j)) + (h<<1);
				}
			}
			return h;
		}
	};
}
template<typename T>
Matrix<T> transpose(Matrix<T> a);
			
template<typename T>
struct Node{
	public:
	Matrix<T> A;
	Matrix<double> U;
	Matrix<double> L;
	vector<array<int,2>> P;
	Node* next;
	Node* back;
	Node(const Matrix<T>& A1,const Matrix<double>& U1,const Matrix<double>& L1,const vector<array<int,2>> P1,Node* next1, Node* back1){
			U = U1;
			A = A1;
			L = L1;
			P = P1;
			next = next1;
			back = back1;
	}
	Node(const Matrix<T>& A1,const Matrix<double>& U1,const Matrix<double>& L1,const vector<array<int,2>> P1){
                        U = U1;
                        A = A1;
                        L = L1;
                        P = P1;
                        next = nullptr;
			back = nullptr;
	}
	Node(){
			next = nullptr;
			back = nullptr;
	}
};
template<typename T>
class LRUcache{
	public:
	unordered_map<Matrix<T>, Node<T>*> map;
	int capacity;
	Node<T>* head;
	Node<T>* tail;
	LRUcache(int capacity1){
		capacity = capacity1;
		vector<array<int,2>> c = {{0,0}};
		head = new Node<T>();
		tail = new Node<T>();
		head->next = tail;
		tail->back = head;

	}

	void deletelast(){
		Node<T>* temp = tail->back;
		temp->next->back = temp->back;
		temp->back->next = temp->next;
		map.erase(temp->A);
		delete temp;
	}
	void isolatenode(Node<T>* x){
                Node<T>* temp = x;
                temp->next->back = temp->back;
                temp->back->next = temp->next;
		x->next = nullptr;
		x->back = nullptr;
        }
	void deletenode(Node<T>* x){
		Node<T>* temp = x;
		temp->next->back = temp->back;
                temp->back->next = temp->next;
		map.erase(temp->A);
		delete temp;
	}
	void insertAfterHead (Node<T>* x){
                Node<T>* temp = head;
                temp->next->back = x;
                x->next = temp->next;
                x->back = temp;
                temp->next = x;

        }
	Node<T>* get(Matrix<T> k){

                Node<T>* x = map[k];
		isolatenode(x);
                insertAfterHead(x);
                return x;
        }
                 
	void put(const Matrix<T> A1,const Matrix<double> U1,const Matrix<double> L1,const vector<array<int,2>> P1){
		if(map.size() == capacity){
			deletelast();
			Node<T>* temp = new Node<T>(A1,U1, L1, P1);
			insertAfterHead( temp);
			map.insert({A1,temp});
		}
		else{
			Node<T>* temp = new Node<T>(A1, U1, L1, P1);
			insertAfterHead(temp);
			map.insert({A1,temp});
			}
	}
};
	
template<typename T>
Matrix<T> MatrixAddition(Matrix<T> a, Matrix<T>  b);
template<typename T>
Matrix<T> MatrixMultiplication(Matrix<T> a, Matrix<T> b);
template<typename T>
Matrix<double> MatrixInverse(Matrix<T> a);
template<typename T>
Matrix<double> EquationSolver(Matrix<T> a, Matrix<T> b);
template<typename T>
Matrix<T> InputMatrix();
template<typename T>
void PrintMatrix(Matrix<T> a);

class LossFunction{
    public:
        double MSE(Matrix<double> pred, Matrix<double> actual);
};
template<typename T>
Matrix<double> Gradient_Descent(Matrix<double> x, Matrix<double> y, double alpha, int epochs);
template<typename T>
class LinearRegressor {  
    public:
        int epoch = 0;
        double alpha = 0; 
        double bias = 0;
        int features;

        LinearRegressor(double learning_rate=0.0001 , int epoch=1000);
        void train(Matrix<double> x, Matrix<double> y);
        Matrix<double> predict(Matrix<double> x, Matrix<double> pred);

    private:
        double gettingValues(vector<double> x, Matrix<double> WandB);
};

		



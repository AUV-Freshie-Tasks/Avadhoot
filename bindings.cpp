#include <pybind11/pybind11.h>
#include "matrix1_01.h"
namespace py = pybind11;

PYBIND11_MODULE(MatrixOperations, m){
	py::class_<Matrix<int>>(m , "Matrix_i")
		.def(py::init<int, int>())
                .def(py::init<>())
	        .def("Set_element", &Matrix<int>::setelement)
		.def("Scalar_multiplication", &Matrix<int>::scalarmult)
		.def("Element", &Matrix<int>::element)
		.def("Columns", &Matrix<int>::C)
		.def("Rows", &Matrix<int>::R);
	m.doc() = "Matrix solver bindings";
	py::class_<Matrix<double>>(m , "Matrix_d")
                .def(py::init<int, int>())
                .def(py::init<>())
                .def("Set_element", &Matrix<double>::setelement)
                .def("Scalar_multiplication", &Matrix<double>::scalarmult)
                .def("Element", &Matrix<double>::element)
                .def("Columns", &Matrix<double>::C)
                .def("Rows", &Matrix<double>::R);

	py::class_<Matrix<float>>(m , "Matrix_f")
                .def(py::init<int, int>())
                .def(py::init<>())
                .def("Set_element", &Matrix<float>::setelement)
                .def("Scalar_multiplication", &Matrix<float>::scalarmult)
                .def("Element", &Matrix<float>::element)
                .def("Columns", &Matrix<float>::C)
                .def("Rows", &Matrix<float>::R);

        m.def("add", &MatrixAddition<double>);
        m.def("add", &MatrixAddition<int>);
        m.def("add", &MatrixAddition<float>);

  
        m.def("mul", &MatrixMultiplication<double>);
        m.def("mul", &MatrixMultiplication<int>);
        m.def("mul", &MatrixMultiplication<float>);

   
        m.def("transpose", &transpose<double>);
        m.def("transpose", &transpose<int>);
        m.def("transpose", &transpose<float>);

    
        m.def("inverse", &MatrixInverse<double>);
        m.def("inverse", &MatrixInverse<float>);

        m.def("solve", &Solve<double>);
        m.def("solve", &Solve<float>);


	m.def("Create_Matrix", &make_matrix<int>);
	m.def("Create_Matrix", &make_matrix<float>);
	m.def("Create_Matrix", &make_matrix<double>);

	m.def("gradient_descent", &Gradient_Descent);


	py::class_<LinearRegressor<double>>(m, "LinearRegressor")
                .def(py::init<double, int>(),
                        py::arg("learning_rate") = 0.0001,
                        py::arg("epochs") = 1000)
                .def("train", &LinearRegressor<double>::train)
                .def("predict", &LinearRegressor<double>::predict);

	py::class_<LossFunction>(m, "LossFunction")
                .def(py::init<>())
                .def("mse", &LossFunction::MSE);
        

}

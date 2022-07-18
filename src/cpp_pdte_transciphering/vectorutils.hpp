/******************************************************	
 * 	Author: Hilder Vitor Lima Pereira 
 *
 * 	e-mail: hilder.vitor@gmail.com
 *
 * 	This file is available in the repository
 * 	https://github.com/hilder-vitor/vector-matrix-utils
 *	Read the LICENSE therein before using it.
 * ****************************************************/

#ifndef ___VECTOR_UTILS_BASICS
#define ___VECTOR_UTILS_BASICS

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace std;

template <typename T1, typename T2>
void operator+=(vector<T1>& u, const vector<T2>& v){
	if (v.size() != u.size())
		throw std::invalid_argument("It is impossible to add vectors of different sizes.");
	unsigned int n = u.size();
	for (unsigned int i = 0; i < n; i++){
		u[i] = u[i] + v[i];
	}
}

template <typename T1, typename T2>
void operator+=(vector<T1>& u, const T2& c){
	unsigned int n = u.size();
	for (unsigned int i = 0; i < n; i++){
		u[i] = u[i] + c;
	}
}

template <typename T1, typename T2>
void operator-=(vector<T1>& u, const vector<T2>& v){
	if (v.size() != u.size())
		throw std::invalid_argument("It is impossible to subtract vectors of different sizes.");
	unsigned int n = u.size();
	for (unsigned int i = 0; i < n; i++){
		u[i] = u[i] - v[i];
	}
}

template <typename T1, typename T2>
void operator-=(vector<T1>& u, const T2& c){
	unsigned int n = u.size();
	for (unsigned int i = 0; i < n; i++){
		u[i] = u[i] - c;
	}
}

template <typename T1, typename T2>
void operator*=(vector<T1>& u, const T2& c){
	unsigned int n = u.size();
	for (unsigned int i = 0; i < n; i++){
		u[i] = u[i] * c;
	}
}

template <typename T1, typename T2>
void operator/=(vector<T1>& u, const T2& m){
	unsigned int n = u.size();
	for (unsigned int i = 0; i < n; i++){
		u[i] = u[i] / m;
	}
}

template <typename T1, typename T2>
vector<T1> operator+(const vector<T1>& u, const vector<T2>& v){
	vector<T1> vec(u);
	vec += v;
	return vec;
}

template <typename T1, typename T2>
vector<T1> operator+(const vector<T1>& u, const T2& c){
	vector<T1> vec(u);
	vec += c;
	return vec;
}

template <typename T1, typename T2>
vector<T1> operator+(const T1& c, const vector<T2>& v){
	return v + c; 
}

template <typename T1, typename T2>
vector<T1> operator-(const vector<T1>& u, const vector<T2>& v){
	vector<T1> vec(u);
	vec -= v;
	return vec;
}

template <typename T1, typename T2>
vector<T1> operator-(const vector<T1>& u, const T2& c){
	vector<T1> vec(u);
	vec -= c;
	return vec;
}

template <typename T1, typename T2>
T1 operator*(const vector<T1>& u, const vector<T2>& v){
	if (u.size() != v.size())
		throw std::invalid_argument("It is impossible to multiply vectors of different sizes.");
	unsigned int n = u.size();
	T1 innerProduct(u[0] * v[0]);
	for (unsigned int i = 1; i < n; i++){
		innerProduct = innerProduct + u[i] * v[i];
	}

	return innerProduct;
}

template <typename T1, typename T2>
vector<T1> operator*(const vector<T1>& u, const T2& c){
	vector<T1> vec(u);
	vec *= c;
	return vec;
}

/* XXX: It is assuming that * is commutative  */
template <typename T1, typename T2>
vector<T1> operator*(const T1& c, const vector<T2>& v){
	return v * c;
}

template <typename T1, typename T2>
vector<T1> product_component_by_component(const vector<T1>& u, const vector<T2>& v){
	if (u.size() != v.size())
		throw std::invalid_argument("It is impossible to multiply vectors of different sizes.");
	unsigned int n = u.size();
	vector<T1> resp;
	for (unsigned int i = 0; i < n; i++){
		resp.push_back(u[i] * v[i]);
	}
	return resp;
}

template <typename T1, typename T2>
vector<T1> operator/(const vector<T1>& u, const T2& m){
	vector<T1> vec(u);
	vec /= m;
	return vec;
}

template <typename ELEMENT>
std::ostream& operator<<(std::ostream& os, const vector<ELEMENT>& u){
	unsigned int lastPosition = u.size() - 1;
	for (unsigned int i = 0; i < lastPosition; i++){
		os << u[i] << ", ";
	}
	os << u[lastPosition];
	return os;
}

#endif

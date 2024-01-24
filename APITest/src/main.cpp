#include <SE.Math.hpp>
using namespace SIByL;

void main() {
  Math::mat4 transform_mat;
  Math::vec3 t, r, s;
  Math::vec3 a = Math::RotationMatrixToEulerAngles(transform_mat);
  Math::Decompose(transform_mat, &t, &r, &s);
  float f = 1.f;
}


//#include <iostream>
//#include <type_traits>
//#include <functional>
//#include <SE.Core.Reflect.hpp>
//#include <SE.Core.Serialize.hpp>
//#include <array>
//
//using namespace SIByL;
//
//struct A : public ISerializable, public IObject {
//  int foo(int i) {
//    std::cout << "call";
//    return i + b;
//  }
//
//  void bar() { std::cout << "bar";
//
//  }
//  SERIALIZE(b);
//  int b = 2;
//};
//
//REGISTER_CLASS(A);
//REGISTER_CLASS_METHOD(A, bar);
//REGISTER_CLASS_METHOD(A, foo);
//
//
//template <typename... Params>
//auto Call(IObject* obj, uintptr_t method, Params&&... params) -> void {
//  using FuncPtr = int (IObject::*)(Params...);
//  auto func = *reinterpret_cast<FuncPtr*>(method);
//  (obj->*func)(std::forward<Params>(params)...);
//}
//
//int main() {
//  std::array<int, 3> test {1, 2, 3};
//  auto [a, b, c] = test;
//
//
//  A obj;  // Create an instance of A
//  IObject* objPtr = &obj;
//  objPtr->call("bar");
//  objPtr->call("foo", 25);
//  int i = objPtr->call_with_return<int>("foo", 125);
//
//  //auto FUNC = [fn = &A::foo](IObject* obj, auto&&... args) -> decltype(auto) {
//  //  
//  //  A* a = static_cast<A*>(obj);
//  //  if (a) {
//  //    return (a->*fn)(std::forward<decltype(args)>(args)...);
//  //  } else {
//  //    std::cout << "Error: Invalid object type" << std::endl;
//  //    return 0;
//  //  }
//  //};
///*  inline auto CLASS_NAME##METHOD_NAME##method =
//      [](CLASS_NAME* obj, auto&&... args) -> decltype(auto) {
//    return (obj->*(&CLASS_NAME::METHOD_NAME))(
//        std::forward<decltype(args)>(args)...);
//  };      */
//  //auto func = [fn](A* a, auto&&... args) -> decltype(auto) {
//  //  return (a->*fn)(std::forward<decltype(args)>(args)...);
//  //};  // Generic lambda with trailing return type
//
//  //IObject* objPtr = &obj;
//  //auto funcPtr = reinterpret_cast<uint64_t>(&FUNC);
//
//  // //call funcPtr
//  //Call(objPtr, funcPtr, 5);
//
//  return 0;
//}
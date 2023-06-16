#pragma once

/*****************************************************
* Singleton Template
*----------------------------------------------------
* For example, given struct A, to use it as a
* singleton class, we could get it by:
*	```
*		Singleton<A>::instance()
*	```
* Rigorously, the ctor of A should be defined in private
* to prevent construction other than singleton.
* And macro SINGLETON provide a convenient shortcut:
*	```
*		SINGLETON(A, {
*		  name = "Jack";
*		  age = 19;
*		});
*	```
* But sometimes you may actually need other constructions
* orther than singleton. In this case it is fine not
* to include the macro in the class definition.
* 
* Sometimes, we may hope to explicityly release the
* memory, especially in some case we need different
* singletons be released in some particular order.
* So it could be useful to do it by:
*	```
*		Singleton<A>::release()
*	```
*****************************************************/

namespace SIByL {
inline namespace utility {
template <class T>
struct Singleton {
  // get the singleton instance
  static   T* instance();
  // explicitly release singleton resource
  static void release();
 private:
  Singleton() {}
  Singleton(Singleton<T>&) {}
  Singleton(Singleton<T>&&) {}
  ~Singleton() {}
  Singleton<T>& operator=(Singleton<T> const) {}
  Singleton<T>& operator=(Singleton<T>&&) {}
 private:
  static T* pinstance;
};

#define SINGLETON(T, CTOR)		    \
 private:							\
  friend struct Singleton<T>;		\
  T() CTOR							\
 public:

template <class T>
T* Singleton<T>::instance() {
  if (pinstance == nullptr) pinstance = new T();
  return pinstance;
}
template <class T>
void Singleton<T>::release() {
  if (pinstance != nullptr) delete pinstance;
}
template <class T>
T* Singleton<T>::pinstance = nullptr;
}  // namespace utility
}  // namespace SIByL
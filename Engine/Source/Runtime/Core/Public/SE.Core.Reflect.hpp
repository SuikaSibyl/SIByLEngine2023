#pragma once
#include <string>
#include <cstdint>
#include <typeinfo>
#include <functional>
#include <string_view>
#include <unordered_map>
#include "SE.Core.Utility.hpp"

namespace SIByL {
inline namespace utility {
template <typename T>
constexpr std::string_view get_raw_name() {
#ifdef _MSC_VER
  return __FUNCSIG__;
#else
  return __PRETTY_FUNCTION__;
#endif
}

template <auto T>
constexpr std::string_view get_raw_name() {
#ifdef _MSC_VER
  return __FUNCSIG__;
#else
  return __PRETTY_FUNCTION__;
#endif
}

template <typename T>
inline constexpr std::string_view type_string() {
  constexpr std::string_view sample = get_raw_name<int>();
  constexpr size_t pos = sample.find("int");
  constexpr std::string_view str = get_raw_name<T>();
  constexpr auto next1 = str.rfind(
      sample[pos + 3]);
#if defined(_MSC_VER)
  constexpr auto s1 = str.substr(pos + 6, next1 - pos - 6);
#else
  constexpr auto s1 = str.substr(pos, next1 - pos);
#endif
  return s1;
}

template <auto T>
inline constexpr std::string_view enum_string() {
  constexpr std::string_view sample = get_raw_name<int>();
  constexpr size_t pos = sample.find("int");
  constexpr std::string_view str = get_raw_name<T>();
  constexpr auto next1 = str.rfind(sample[pos + 3]);
#if defined(__clang__) || defined(_MSC_VER)
  constexpr auto s1 = str.substr(pos, next1 - pos);
#else
  constexpr auto s1 = str.substr(pos + 5, next1 - pos - 5);
#endif
  return s1;
}

struct ClassField {
  ClassField() : name(""), type(""), offset(0){}
  ClassField(std::string const& name, std::string const& type, int64_t offset)
      : name(name), type(type), offset(offset){}
  ~ClassField() = default;
  auto getName() const noexcept->std::string const& { return name; }
  auto getType() const noexcept->std::string const& { return type; }
  auto getOffset() const noexcept -> int64_t { return offset; }
 private:
  std::string name;
  std::string type;
  int64_t offset;
};

struct ClassMethod {
  ClassMethod() : name(""), method(0), offset(0) {}
  ClassMethod(std::string const& name, uintptr_t method, int64_t offset)
      : name(name), method(method), offset(offset) {}
  ~ClassMethod() = default;
  auto getName() const noexcept -> std::string const& { return name; }
  auto getMethod() const noexcept -> uintptr_t { return method; }
  auto getOffset() const noexcept -> int64_t { return offset; }
 private:
  std::string name;
  uintptr_t method;
  int64_t offset;
};

struct IObject {
  IObject() = default;
  virtual ~IObject() = default;
  // Get field info and field data
  auto get_field_count() const noexcept->size_t;
  auto get_field(size_t pos) const noexcept->ClassField*;
  auto get_field(std::string const& fieldName) const noexcept->ClassField*;
  template<class T>
  auto get(std::string const& fieldName) const noexcept -> T&;
  template <class T>
  auto set(std::string const& fieldName, T const& value) noexcept -> void;
  auto call(std::string const& methodName) const noexcept -> void;
  template <class... Params>
  auto call(std::string const& methodName, Params&&... params) const noexcept -> void;
  template <class R, class... Params>
  auto call_with_return(std::string const& methodName, Params&&... params) const noexcept -> R;
};

using create_object = std::function<IObject*()>;

struct ClassFactory {
  SINGLETON(ClassFactory, {});
  auto register_class(std::string const& name, create_object method) noexcept -> void;
  auto create_class(std::string const& name) noexcept -> IObject*;

  auto register_class_filed(std::string const& cname, std::string const& fname,
                            std::string const& ftype, int64_t offset) noexcept -> void;
  auto get_field_count(std::string const& name) noexcept -> size_t;
  auto get_field(std::string const& name, size_t pos) noexcept -> ClassField*;
  auto get_field(std::string const& cname, std::string const& fname) noexcept -> ClassField*;

  auto register_class_method(std::string const& cname, std::string const& mname,
                             uintptr_t ptr, int64_t offset) noexcept -> void;
  auto get_method_count(std::string const& name) noexcept -> size_t;
  auto get_method(std::string const& name, size_t pos) noexcept -> ClassMethod*;
  auto get_method(std::string const& cname, std::string const& mname) noexcept
      -> ClassMethod*;
 private:
  std::unordered_map<std::string, create_object> classmap;
  std::unordered_map<std::string, std::vector<ClassField>> classfields;
  std::unordered_map<std::string, std::vector<ClassMethod>> classmethods;
};

struct ClassRegister {
  ClassRegister(std::string const& name, create_object method) {
    ClassFactory* factory = Singleton<ClassFactory>::instance();
    factory->register_class(name, method);
  }

  ClassRegister(std::string const& name, std::string const& fieldnaame,
                std::string const& type, int64_t offset) {
    ClassFactory* factory = Singleton<ClassFactory>::instance();
    factory->register_class_filed(name, fieldnaame, type, offset);
  }

  ClassRegister(std::string const& name, std::string const& methodname,
                uintptr_t ptr, int64_t offset) {
    ClassFactory* factory = Singleton<ClassFactory>::instance();
    factory->register_class_method(name, methodname, ptr, offset);
  }
};

#define TYPE_NAME(CLASS_NAME) typeid(CLASS_NAME).name()

#define REGISTER_CLASS(CLASS_NAME)                          \
  inline ClassRegister classClassRegister##CLASS_NAME(      \
      typeid(CLASS_NAME).name(),                            \
      []() -> IObject* { return new CLASS_NAME(); })

#define REGISTER_CLASS_FIELD(CLASS_NAME, FIELD_NAME)                \
  inline CLASS_NAME CLASS_NAME##FIELD_NAME;                         \
  inline ClassRegister classClassRegister##CLASS_NAME##FIELD_NAME(  \
      typeid(CLASS_NAME).name(), #FIELD_NAME,                       \
      typeid(CLASS_NAME::FIELD_NAME).name(),                        \
      (int64_t)(&(CLASS_NAME##FIELD_NAME.FIELD_NAME)) -             \
      (int64_t)((IObject*)& CLASS_NAME##FIELD_NAME))

#define REGISTER_CLASS_METHOD(CLASS_NAME, METHOD_NAME)                      \
  inline auto CLASS_NAME##METHOD_NAME##method =                             \
      [fn = &CLASS_NAME::METHOD_NAME](IObject* obj,                         \
                                    auto&&... args) -> decltype(auto) {     \
        CLASS_NAME* x = static_cast<CLASS_NAME*>(obj);                      \
        return (x->*fn)(std::forward<decltype(args)>(args)...);             \
      };                                                                    \
  inline ClassRegister classClassRegister##CLASS_NAME##METHOD_NAME(         \
      typeid(CLASS_NAME).name(), #METHOD_NAME,                              \
      (uintptr_t)(&CLASS_NAME##METHOD_NAME##method),                        \
      (int64_t(static_cast<IObject*>(                                       \
        reinterpret_cast<CLASS_NAME*>(0x10000000))) - int64_t(0x10000000))  \
  )

template <class T>
auto IObject::get(std::string const& fieldName) const noexcept -> T& {
  ClassFactory* factory = Singleton<ClassFactory>::instance();
  ClassField* field = get_field(fieldName);
  int64_t offset = field->getOffset();
  return *(T*)((unsigned char*)(this) + offset);
}
template <class T>
auto IObject::set(std::string const& fieldName, T const& value) noexcept
    -> void {
  ClassFactory* factory = Singleton<ClassFactory>::instance();
  ClassField* field = get_field(fieldName);
  int64_t offset = field->getOffset();
  *(T*)((unsigned char*)(this) + offset) = value;
}
template <class... Params>
auto IObject::call(std::string const& methodName, Params&&... params) const noexcept -> void {
  ClassFactory* factory = Singleton<ClassFactory>::instance();
  ClassMethod* method = factory->get_method(typeid(*this).name(), methodName);
  using FuncPtr = void (IObject::*)(Params...);
  auto func = *reinterpret_cast<FuncPtr*>(method->getMethod());
  ((IObject*)this->*func)(std::forward<Params>(params)...);
}
template <class R, class... Params>
auto IObject::call_with_return(std::string const& methodName,
                               Params&&... params) const noexcept -> R {
  ClassFactory* factory = Singleton<ClassFactory>::instance();
  ClassMethod* method = factory->get_method(typeid(*this).name(), methodName);
  using FuncPtr = R (IObject::*)(Params...);
  auto func = *reinterpret_cast<FuncPtr*>(method->getMethod());
  return ((IObject*)this->*func)(std::forward<Params>(params)...);
}
}  // namespace utility
}  // namespace SIByL
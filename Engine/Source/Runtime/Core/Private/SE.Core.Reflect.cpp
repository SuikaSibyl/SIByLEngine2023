#include "../Public/SE.Core.Reflect.hpp"

namespace SIByL {
inline namespace utility {
auto IObject::get_field_count() const noexcept -> size_t {
  ClassFactory* factory = Singleton<ClassFactory>::instance();
  return factory->get_field_count(typeid(*this).name());
}
auto IObject::get_field(size_t pos) const noexcept->ClassField* {
  ClassFactory* factory = Singleton<ClassFactory>::instance();
  return factory->get_field(typeid(*this).name(), pos);
}
auto IObject::get_field(std::string const& fieldName) const noexcept
    -> ClassField* {
  ClassFactory* factory = Singleton<ClassFactory>::instance();
  return factory->get_field(typeid(*this).name(), fieldName);
}
auto IObject::call(std::string const& methodName) const noexcept -> void {
  ClassFactory* factory = Singleton<ClassFactory>::instance();
  ClassMethod* method = factory->get_method(typeid(*this).name(), methodName);
  using FuncPtr = void (IObject::*)();
  auto func = *reinterpret_cast<FuncPtr*>(method->getMethod());
  ((IObject*)this->*func)();
}

auto ClassFactory::register_class(std::string const& name,
        create_object method) noexcept -> void {
  classmap[name] = method;
}

auto ClassFactory::create_class(std::string const& name) noexcept -> IObject* {
  auto it = classmap.find(name);
  if (it != classmap.end()) return it->second();
  else return nullptr;
}

auto ClassFactory::register_class_filed(std::string const& cname,
    std::string const& fname,
    std::string const& ftype,
    int64_t offset) noexcept -> void {
  classfields[cname].push_back(ClassField{fname, ftype, offset});
  }

auto ClassFactory::get_field_count(std::string const& name) noexcept -> size_t {
  auto iter = classfields.find(name);
  if (iter == classfields.end()) return 0;
  else return iter->second.size();
}

auto ClassFactory::get_field(std::string const& name, size_t pos) noexcept
-> ClassField* {
  auto iter = classfields.find(name);
  if (iter == classfields.end()) return nullptr;
  else if (iter->second.size()<=pos) return nullptr;
  else return &iter->second[pos];
}

auto ClassFactory::get_field(std::string const& cname, std::string const& fname) noexcept -> ClassField* {
  auto iter = classfields.find(cname);
  if (iter == classfields.end()) return nullptr;
  else {
    for (auto& field : iter->second) {
      if (field.getName() == fname) return &field;
    }
    return nullptr;
  }
}

auto ClassFactory::register_class_method(std::string const& cname,
                                         std::string const& mname,
                                         uintptr_t ptr, int64_t offset) noexcept
    -> void {
  classmethods[cname].emplace_back(ClassMethod{mname, ptr, offset});
}

auto ClassFactory::get_method_count(std::string const& name) noexcept -> size_t {
  auto iter = classmethods.find(name);
  if (iter == classmethods.end()) return 0;
  else return iter->second.size();
}

auto ClassFactory::get_method(std::string const& name, size_t pos) noexcept
-> ClassMethod* {
  auto iter = classmethods.find(name);
  if (iter == classmethods.end()) return nullptr;
  else if (iter->second.size()<=pos) return nullptr;
  else return &iter->second[pos];
}

auto ClassFactory::get_method(std::string const& cname, std::string const& mname) noexcept
-> ClassMethod* {
  auto iter = classmethods.find(cname);
  if (iter == classmethods.end()) return nullptr;
  else {
    for (auto& field : iter->second) {
      if (field.getName() == mname) return &field;
    }
    return nullptr;
  }
}

}  // namespace utility
}  // namespace SIByL
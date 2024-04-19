namespace se {
template<class Type>
struct CPPType{
  Type value;
  CPPType() = default;
  CPPType(Type const& v) :value(v) {}
  auto get() noexcept -> Type& { return value; }
  auto set(Type const& v) noexcept -> void { value = v; }
};
}
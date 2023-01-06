module;
#include <set>
#include <vector>
#include <array>
export module SE.Core.ECS:SparseSet;

namespace SIByL::Core 
{
	export template<int MAX_CAPACITY> struct SparseSet {
		/** insert a new element into set */
		auto insert(int x) noexcept -> void;
		/** delete an element */
		auto remove(int x) noexcept -> void;
		/** search an element */
		auto search(int x) noexcept -> int;
		/** get the set of sparse index */
		auto getSparseSet() noexcept -> std::set<uint64_t>;
		/** sparse array */
		std::array<int, MAX_CAPACITY> sparse;
		/** dense array */
		std::array<int, MAX_CAPACITY> dense;
		/** element live in dense array */
		int livingElementCount = 0;
	};

	template<int MAX_CAPACITY>
	auto SparseSet<MAX_CAPACITY>::insert(int x) noexcept -> void {
		if (x > MAX_CAPACITY) return;
		if (livingElementCount >= MAX_CAPACITY) return;
		if (search(x) != -1) return;
		dense[livingElementCount] = x;
		sparse[x] = livingElementCount;
		++livingElementCount;
	}

	template<int MAX_CAPACITY>
	auto SparseSet<MAX_CAPACITY>::remove(int x) noexcept -> void {
		if (search(x) == -1) return;
		int tmp = dense[livingElementCount - 1];
		dense[sparse[x]] = tmp;
		sparse[tmp] = sparse[x];
		--livingElementCount;
	}

	template<int MAX_CAPACITY>
	auto SparseSet<MAX_CAPACITY>::search(int x) noexcept -> int {
		if (x > MAX_CAPACITY) return -1;
		if (sparse[x] < livingElementCount && dense[sparse[x]] == x)
			return sparse[x];
		return -1;
	}

	template<int MAX_CAPACITY>
	auto SparseSet<MAX_CAPACITY>::getSparseSet() noexcept -> std::set<uint64_t> {
		return std::set<uint64_t>(dense.begin(), dense.begin() + livingElementCount);
	}
}
module;
#include <set>
#include <stack>
#include <queue>
#include <optional>
#include <functional>
#include <unordered_map>
#include <vector>
#include <string>
#include <variant>
export module SE.RDG:DAG;
import SE.Core.Log;
import SE.Utility;
import SE.Math.Geometric;
import SE.RHI;
import SE.GFX;

namespace SIByL::RDG
{
	export struct DAG {

		auto addEdge(uint32_t src, uint32_t dst) noexcept -> void {
			adj[src].insert(dst);
		}

		auto reverse() const noexcept -> DAG {
			DAG r;
			for (auto pair : adj) {
				uint32_t iv = pair.first;
				for (uint32_t const& iw : pair.second)
					r.addEdge(iw, iv);
			}
			/*for (int iv = 0; iv < v; ++iv)
				for (int iw : adj[iv])*/
					//r.addEdge(iw, iv);
			return r;
		}

		std::unordered_map<uint32_t, std::set<uint32_t>> adj;
	};

	export inline auto flatten_bfs(DAG const& g, size_t output) noexcept -> std::optional<std::vector<size_t>> {
		DAG forward = g.reverse();
		DAG reverse = g;
		std::stack<size_t> revList;
		std::queue<size_t> waitingList;
		auto takeNode = [&](size_t node) noexcept -> void {
			revList.push(node);
			for (auto& pair : reverse.adj)
				pair.second.erase(node);
			reverse.adj.erase(node);
		};

		waitingList.push(output);
		while (!waitingList.empty()) {
			size_t front = waitingList.front();
			waitingList.pop();
			takeNode(front);
			for (auto& pair : reverse.adj)
				if (pair.second.size() == 0) {
					waitingList.push(pair.first);
					break;
				}
		}
		std::vector<size_t> flattened;
		while (!revList.empty()) {
			flattened.push_back(revList.top());
			revList.pop();
		}
		return flattened;
	}
}
#include <ECS/SE.Core.ECS.hpp>

namespace SIByL::Core {
EntityManager* EntityManager::singleton = nullptr;
ComponentManager* ComponentManager::singleton = nullptr;

auto EntityManager::get() noexcept -> EntityManager* { return singleton; }
auto ComponentManager::get() noexcept -> ComponentManager* { return singleton; }

auto EntityManager::startUp() noexcept -> void {
  // initialize the queue with all possible entity IDs
  for (EntityHandle entity = 0; entity < MAX_ENTITIES; ++entity)
    availableEntities.emplace(entity);
  singleton = this;
}

auto EntityManager::shutDown() noexcept -> void { availableEntities = {}; }

auto ComponentManager::startUp() noexcept -> void { singleton = this; }

auto EntityManager::createEntity() noexcept -> Entity {
  if (livingEntityCount >= MAX_ENTITIES) {
    LogManager::Error("ECS :: Entity creation exceed pool size.");
    return Entity(NULL_ENTITY);
  }
  // take an ID from the queue.
  ++livingEntityCount;
  EntityHandle id = availableEntities.front();
  signatures[id] = 0;
  availableEntities.pop();
  return Entity(id);
}

auto EntityManager::destroyEntity(EntityHandle id) noexcept -> void {
  // push an ID back to the queue.
  --livingEntityCount;
  availableEntities.push(id);
  ComponentManager::get()->destroyEntity(id);
}

auto EntityManager::getSignature(EntityHandle id) noexcept -> Signature& {
  return signatures[id];
}
}
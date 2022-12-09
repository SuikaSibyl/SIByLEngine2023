module;
#include <typeinfo>
export module SE.Application:Root;
import SE.Core.Log;
import SE.Core.Memory;
import SE.Core.ECS;
import SE.Core.Resource;
import SE.GFX.Core;

namespace SIByL::Application
{
	export struct Root {
		Root();
		~Root();

		Core::MemoryManager		gMemManager;
		Core::LogManager		gLogManager;
		Core::EntityManager		gEntityManager;
		Core::ComponentManager	gComponentManager;
		Core::ResourceManager	gResourceManager;
		GFX::GFXManager			gfxManager;
	};

	Root::Root() {
		gMemManager.startUp();
		gLogManager.startUp();
		gEntityManager.startUp();
		gComponentManager.startUp();
		gResourceManager.startUp();
		gfxManager.startUp();
	}

	Root::~Root() {
		gfxManager.shutDown();
		gResourceManager.shutDown();
		gComponentManager.shutDown();
		gEntityManager.shutDown();
		gLogManager.shutDown();
		gMemManager.shutDown();
	}

}
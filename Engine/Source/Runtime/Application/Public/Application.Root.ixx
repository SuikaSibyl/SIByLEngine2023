module;
#include <typeinfo>
export module Application.Root;
import Core.Log;
import Core.Memory;
import Core.ECS;
import Core.Resource.RuntimeManage;
import GFX.GFXManager;

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
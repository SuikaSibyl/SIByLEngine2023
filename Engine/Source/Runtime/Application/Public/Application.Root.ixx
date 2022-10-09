export module Application.Root;
import Core.Log;
import Core.Memory;
import Core.ECS;

namespace SIByL::Application
{
	export struct Root {
		Root();
		~Root();

		Core::MemoryManager		gMemManager;
		Core::LogManager		gLogManager;
		Core::EntityManager		gEntityManager;
		Core::ComponentManager	gComponentManager;
	};

	Root::Root() {
		gMemManager.startUp();
		gLogManager.startUp();
		gEntityManager.startUp();
		gComponentManager.startUp();
	}

	Root::~Root() {
		gComponentManager.shutDown();
		gEntityManager.shutDown();
		gLogManager.shutDown();
		gMemManager.shutDown();
	}

}
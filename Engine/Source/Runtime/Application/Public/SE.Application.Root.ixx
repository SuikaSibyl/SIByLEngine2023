module;
#include <typeinfo>
export module SE.Application:Root;
import SE.Core.Log;
import SE.Core.Memory;
import SE.Core.ECS;
import SE.Core.Resource;
import SE.GFX;
import SE.Video;

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
		GFX::GFXManager			gGfxManager;
	};

	Root::Root() {
		gMemManager.startUp();
		gLogManager.startUp();
		gEntityManager.startUp();
		gComponentManager.startUp();
		gResourceManager.startUp();
		// ext insertion
		gGfxManager.addExt<GFX::VideExtension>(GFX::Ext::VideoClip);
		gGfxManager.startUp();
	}

	Root::~Root() {
		gGfxManager.shutDown();
		gResourceManager.shutDown();
		gComponentManager.shutDown();
		gEntityManager.shutDown();
		gLogManager.shutDown();
		gMemManager.shutDown();
	}

}
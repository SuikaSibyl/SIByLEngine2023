#include <random>
#include <string>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <thread>
#include <unordered_map>
#include <yaml-cpp/yaml.h>
#include <yaml-cpp/node/node.h>
#include "SE.Core.Resource.hpp"
#include <Memory/SE.Core.Memory.hpp>
#include <IO/SE.Core.IO.hpp>
#include <Print/SE.Core.Log.hpp>

namespace SIByL::Core::Impl {
    //Google CityHash
    using namespace std;

    typedef std::pair<uint64_t, uint64_t> uint128_t;
#define STATIC_INLINE static inline

    STATIC_INLINE uint64_t Uint128Low64(const uint128_t& x) { return x.first; }
    STATIC_INLINE uint64_t Uint128High64(const uint128_t& x) { return x.second; }

    static const uint64_t kSeed0 = 1234567;
    static const uint64_t kSeed1 = 0xc3a5c85c97cb3127ULL;
    static const uint128_t kSeed128(kSeed0, kSeed1);

    STATIC_INLINE uint64_t UNALIGNED_LOAD64(const char* p) {
        uint64_t result;
        memcpy(&result, p, sizeof(result));
        return result;
    }

    STATIC_INLINE uint32_t UNALIGNED_LOAD32(const char* p) {
        uint32_t result;
        memcpy(&result, p, sizeof(result));
        return result;
    }

#define uint32_in_expected_order(x) (x)
#define uint64_in_expected_order(x) (x)

    STATIC_INLINE uint32_t bswap_32(const uint32_t x) {
        uint32_t y = x;
        for (size_t i = 0; i < sizeof(uint32_t) >> 1; ++i) {
            uint32_t d = static_cast<uint32_t>(sizeof(uint32_t) - i - 1);
            uint32_t mh = ((uint32_t)0xff) << (d << 3);
            uint32_t ml = ((uint32_t)0xff) << (i << 3);
            uint32_t h = x & mh;
            uint32_t l = x & ml;
            uint64_t t = (l << ((d - i) << 3)) | (h >> ((d - i) << 3));
            y = static_cast<uint32_t>(t | (y & ~(mh | ml)));
        }
        return y;
    }

    STATIC_INLINE uint64_t bswap_64(const uint64_t x) {
        uint64_t y = x;
        for (size_t i = 0; i < sizeof(uint64_t) >> 1; ++i) {
            uint64_t d = sizeof(uint64_t) - i - 1;
            uint64_t mh = ((uint64_t)0xff) << (d << 3);
            uint64_t ml = ((uint64_t)0xff) << (i << 3);
            uint64_t h = x & mh;
            uint64_t l = x & ml;
            uint64_t t = (l << ((d - i) << 3)) | (h >> ((d - i) << 3));
            y = t | (y & ~(mh | ml));
        }
        return y;
    }

    // Hash 128 input bits down to 64 bits of output.
    // This is intended to be a reasonably good hash function.
    STATIC_INLINE uint64_t Hash128to64(const uint128_t& x) {
        // Murmur-inspired hashing.
        const uint64_t kMul = 0x9ddfea08eb382d69ULL;
        uint64_t a = (Uint128Low64(x) ^ Uint128High64(x)) * kMul;
        a ^= (a >> 47);
        uint64_t b = (Uint128High64(x) ^ a) * kMul;
        b ^= (b >> 47);
        b *= kMul;
        return b;
    }

    STATIC_INLINE uint64_t Fetch64(const char* p) {
        return uint64_in_expected_order(UNALIGNED_LOAD64(p));
    }

    STATIC_INLINE uint32_t Fetch32(const char* p) {
        return uint32_in_expected_order(UNALIGNED_LOAD32(p));
    }

    // Bitwise right rotate.  Normally this will compile to a single
    // instruction, especially if the shift is a manifest constant.
    STATIC_INLINE uint64_t Rotate(uint64_t val, int shift) {
        // Avoid shifting by 64: doing so yields an undefined result.
        return shift == 0 ? val : ((val >> shift) | (val << (64 - shift)));
    }

    STATIC_INLINE uint64_t ShiftMix(uint64_t val) {
        return val ^ (val >> 47);
    }

    STATIC_INLINE uint64_t HashLen16(uint64_t u, uint64_t v) {
        return Hash128to64(uint128_t(u, v));
    }

    STATIC_INLINE uint64_t HashLen16(uint64_t u, uint64_t v, uint64_t mul) {
        // Murmur-inspired hashing.
        uint64_t a = (u ^ v) * mul;
        a ^= (a >> 47);
        uint64_t b = (v ^ a) * mul;
        b ^= (b >> 47);
        b *= mul;
        return b;
    }

    // Some primes between 2^63 and 2^64 for various uses.
    static const uint64_t k0 = 0xc3a5c85c97cb3127ULL;
    static const uint64_t k1 = 0xb492b66fbe98f273ULL;
    static const uint64_t k2 = 0x9ae16a3b2f90404fULL;

    STATIC_INLINE uint64_t HashLen0to16(const char* s, size_t len) {
        if (len >= 8) {
            uint64_t mul = k2 + len * 2;
            uint64_t a = Fetch64(s) + k2;
            uint64_t b = Fetch64(s + len - 8);
            uint64_t c = Rotate(b, 37) * mul + a;
            uint64_t d = (Rotate(a, 25) + b) * mul;
            return HashLen16(c, d, mul);
        }
        if (len >= 4) {
            uint64_t mul = k2 + len * 2;
            uint64_t a = Fetch32(s);
            return HashLen16(len + (a << 3), Fetch32(s + len - 4), mul);
        }
        if (len > 0) {
            uint8_t a = s[0];
            uint8_t b = s[len >> 1];
            uint8_t c = s[len - 1];
            uint32_t y = static_cast<uint32_t>(a) + (static_cast<uint32_t>(b) << 8);
            uint32_t z = static_cast<uint32_t>(len) + (static_cast<uint32_t>(c) << 2);
            return ShiftMix(y * k2 ^ z * k0) * k2;
        }
        return k2;
    }

    // This probably works well for 16-byte strings as well, but it may be overkill
    // in that case.
    STATIC_INLINE uint64_t HashLen17to32(const char* s, size_t len) {
        uint64_t mul = k2 + len * 2;
        uint64_t a = Fetch64(s) * k1;
        uint64_t b = Fetch64(s + 8);
        uint64_t c = Fetch64(s + len - 8) * mul;
        uint64_t d = Fetch64(s + len - 16) * k2;
        return HashLen16(Rotate(a + b, 43) + Rotate(c, 30) + d,
            a + Rotate(b + k2, 18) + c, mul);
    }

    // Return a 16-byte hash for 48 bytes.  Quick and dirty.
    // Callers do best to use "random-looking" values for a and b.
    STATIC_INLINE pair<uint64_t, uint64_t> WeakHashLen32WithSeeds(
        uint64_t w, uint64_t x, uint64_t y, uint64_t z, uint64_t a, uint64_t b) {
        a += w;
        b = Rotate(b + a + z, 21);
        uint64_t c = a;
        a += x;
        a += y;
        b += Rotate(a, 44);
        return make_pair(a + z, b + c);
    }

    // Return a 16-byte hash for s[0] ... s[31], a, and b.  Quick and dirty.
    STATIC_INLINE pair<uint64_t, uint64_t> WeakHashLen32WithSeeds(
        const char* s, uint64_t a, uint64_t b) {
        return WeakHashLen32WithSeeds(Fetch64(s),
            Fetch64(s + 8),
            Fetch64(s + 16),
            Fetch64(s + 24),
            a,
            b);
    }

    // Return an 8-byte hash for 33 to 64 bytes.
    STATIC_INLINE uint64_t HashLen33to64(const char* s, size_t len) {
        uint64_t mul = k2 + len * 2;
        uint64_t a = Fetch64(s) * k2;
        uint64_t b = Fetch64(s + 8);
        uint64_t c = Fetch64(s + len - 24);
        uint64_t d = Fetch64(s + len - 32);
        uint64_t e = Fetch64(s + 16) * k2;
        uint64_t f = Fetch64(s + 24) * 9;
        uint64_t g = Fetch64(s + len - 8);
        uint64_t h = Fetch64(s + len - 16) * mul;
        uint64_t u = Rotate(a + g, 43) + (Rotate(b, 30) + c) * 9;
        uint64_t v = ((a + g) ^ d) + f + 1;
        uint64_t w = bswap_64((u + v) * mul) + h;
        uint64_t x = Rotate(e + f, 42) + c;
        uint64_t y = (bswap_64((v + w) * mul) + g) * mul;
        uint64_t z = e + f + c;
        a = bswap_64((x + z) * mul + y) + b;
        b = ShiftMix((z + a) * mul + d + h) * mul;
        return b + x;
    }

    uint64_t CityHash64(const char* s, size_t len) {
        if (len <= 32) {
            if (len <= 16) {
                return HashLen0to16(s, len);
            }
            else {
                return HashLen17to32(s, len);
            }
        }
        else if (len <= 64) {
            return HashLen33to64(s, len);
        }

        // For strings over 64 bytes we hash the end first, and then as we
        // loop we keep 56 bytes of state: v, w, x, y, and z.
        uint64_t x = Fetch64(s + len - 40);
        uint64_t y = Fetch64(s + len - 16) + Fetch64(s + len - 56);
        uint64_t z = HashLen16(Fetch64(s + len - 48) + len, Fetch64(s + len - 24));
        pair<uint64_t, uint64_t> v = WeakHashLen32WithSeeds(s + len - 64, len, z);
        pair<uint64_t, uint64_t> w = WeakHashLen32WithSeeds(s + len - 32, y + k1, x);
        x = x * k1 + Fetch64(s);

        // Decrease len to the nearest multiple of 64, and operate on 64-byte chunks.
        len = (len - 1) & ~static_cast<size_t>(63);
        do {
            x = Rotate(x + y + v.first + Fetch64(s + 8), 37) * k1;
            y = Rotate(y + v.second + Fetch64(s + 48), 42) * k1;
            x ^= w.second;
            y += v.first + Fetch64(s + 40);
            z = Rotate(z + w.first, 33) * k1;
            v = WeakHashLen32WithSeeds(s, v.second * k1, x + w.first);
            w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + Fetch64(s + 16));
            std::swap(z, x);
            s += 64;
            len -= 64;
        } while (len != 0);
        return HashLen16(HashLen16(v.first, w.first) + ShiftMix(y) * k1 + z,
            HashLen16(v.second, w.second) + x);
    }

    uint64_t CityHash64WithSeeds(const char* s, size_t len,
        uint64_t seed0, uint64_t seed1) {
        return HashLen16(CityHash64(s, len) - seed0, seed1);
    }

    uint64_t CityHash64WithSeed(const char* s, size_t len, uint64_t seed) {
        return CityHash64WithSeeds(s, len, k2, seed);
    }
}  // namespace SIByL::Core::Impl

    namespace SIByL::Core {

    auto hashUID(char const* path) noexcept -> uint64_t {
        return Impl::CityHash64(path, strlen(path));
    }

    auto requestORID() noexcept -> ORID {
        static std::default_random_engine e;
        static std::uniform_int_distribution<uint64_t> u(0, 0X3FFFFF);

        ORID id = 0;
        time_t now = time(0);
        tm ltm;
        localtime_s(&ltm, &now);
        id += (uint64_t(ltm.tm_year - 100) & 0xFF) << 56;
        id += (uint64_t(ltm.tm_mon) & 0xF) << 52;
        id += (uint64_t(ltm.tm_mday) & 0x1F) << 47;
        id += (uint64_t(ltm.tm_hour) & 0x1F) << 42;
        id += (uint64_t(ltm.tm_min) & 0x3F) << 36;
        id += (uint64_t(ltm.tm_sec) & 0x3F) << 30;

        std::thread::id tid = std::this_thread::get_id();
        unsigned int nId = *(unsigned int*)((char*)&tid);
        id += (uint64_t(nId) & 0xFF) << 22;
        id += u(e);
        return id;
    }

    auto ResourceDatabase::registerResource(Core::ORID orid,
                                            Core::GUID guid) noexcept -> void {
        mapper[orid] = guid;
    }

    auto ResourceDatabase::findResource(Core::ORID orid) noexcept
        -> Core::GUID {
        auto iter = mapper.find(orid);
        if (iter == mapper.end())
            return INVALID_ORID;
        else
            return iter->second;
    }

    auto ResourceDatabase::findResourcePath(char const* path_c) noexcept
        -> Core::ORID {
        std::filesystem::path path(path_c);
        std::filesystem::path current_path = std::filesystem::current_path();
        std::filesystem::path relative_path =
            std::filesystem::relative(path, current_path);
        auto iter = resource_mapper.find(relative_path.string());
        if (iter == resource_mapper.end()) {
            return INVALID_ORID;
        } else
            return iter->second;
    }

    auto ResourceDatabase::mapResourcePath(char const* path_c) noexcept
        -> Core::ORID {
        std::filesystem::path path(path_c);
        std::filesystem::path current_path = std::filesystem::current_path();
        std::filesystem::path relative_path =
            std::filesystem::relative(path, current_path);
        auto iter = resource_mapper.find(relative_path.string());
        if (iter == resource_mapper.end()) {
            Core::ORID orid = requestORID();
            ;
            resource_mapper[relative_path.string()] = orid;
            return orid;
        } else
            return iter->second;
    }

    auto ResourceDatabase::serialize() noexcept -> void {
        std::filesystem::path path = "./bin/.adb";
        YAML::Emitter out;
        out << YAML::BeginMap;
        out << YAML::Key << "Prefix" << YAML::Value << "AssetDatabase";
        out << YAML::Key << "Entries" << YAML::Value << YAML::BeginSeq;
        for (auto& [name, ORID] : resource_mapper) {
            out << YAML::BeginMap;
            out << YAML::Key << "PATH" << YAML::Value << name;
            out << YAML::Key << "ORID" << YAML::Value << ORID;
            out << YAML::EndMap;
        }
        out << YAML::EndSeq;
        out << YAML::Key << "End" << YAML::Value << "TRUE";
        Core::Buffer adb_proxy;
        adb_proxy.data = (void*)out.c_str();
        adb_proxy.size = out.size();
        Core::syncWriteFile(path.string().c_str(), adb_proxy);
        adb_proxy.data = nullptr;
    }

    auto ResourceDatabase::deserialize() noexcept -> void {
        std::filesystem::path path = "./bin/.adb";
        Core::Buffer adb_proxy;
        Core::syncReadFile(path.string().c_str(), adb_proxy);
        if (adb_proxy.size != 0) {
            YAML::NodeAoS data =
                YAML::Load(reinterpret_cast<char*>(adb_proxy.data));
            // check scene name
            if (!data["Prefix"]) {
                Core::LogManager::Error(
                    "GFX :: Asset Database not found when deserializing {0}");
                return;
            }
            auto entries = data["Entries"];
            for (auto node : entries) {
                resource_mapper[node["PATH"].as<std::string>()] =
                    node["ORID"].as<Core::ORID>();
            }
        }
    }

    ResourceManager* ResourceManager::singleton = nullptr;

    auto ResourceManager::startUp() noexcept -> void {
        singleton = this;
        database.deserialize();
    }

    auto ResourceManager::shutDown() noexcept -> void { database.serialize(); }

    auto ResourceManager::get() noexcept -> ResourceManager* {
        return singleton;
    }

    auto ResourceManager::clear() noexcept -> void { resourcePools.clear(); }
    }  // namespace SIByL::Core
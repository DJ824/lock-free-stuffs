#pragma once

#include "sync.h"

#include <cstddef>
#include <memory>
#include <new>
#include <utility>

namespace support {

class HugePages {
    unsigned char* cur_{nullptr};
    unsigned char* end_{nullptr};
    unsigned char* beg_{nullptr};

public:
    using WarnFn = void();
    static WarnFn* warn_no_1GB_pages;
    static WarnFn* warn_no_2MB_pages;

    struct Deleter {
        HugePages* hp_{nullptr};

        template <class T>
        void operator()(T* p) const {
            hp_->destroy(p);
        }
    };

    enum Type {
        PAGE_DEFAULT = 0,
        PAGE_2MB = 21,
        PAGE_1GB = 30,
    };

    HugePages(Type type, size_t total_size);
    ~HugePages() noexcept;

    HugePages(HugePages const&) = delete;
    HugePages& operator=(HugePages const&) = delete;

    HugePages(HugePages&& other) noexcept;
    HugePages& operator=(HugePages&& other) noexcept;

    void reset() noexcept;
    bool empty() const noexcept;
    size_t used() const noexcept;

    void* allocate(size_t size, std::nothrow_t) noexcept;
    void* allocate(size_t size);
    void deallocate(void* p, size_t size) noexcept;

    template <class T, class... Args>
    std::unique_ptr<T, Deleter> create_unique_ptr(Args&&... args) {
        return std::unique_ptr<T, Deleter>{
            new (allocate(sizeof(T))) T{std::forward<Args>(args)...},
            Deleter{this}
        };
    }

    template <class T, class... Args>
    std::unique_ptr<T, Deleter> create_unique_ptr(NoContext, Args&&... args) {
        return std::unique_ptr<T, Deleter>{
            new (allocate(sizeof(T))) T{std::forward<Args>(args)...},
            Deleter{this}
        };
    }

    template <class T>
    void destroy(T* p) {
        void* raw = p;
        p->~T();
        deallocate(raw, sizeof(T));
    }
};

struct HugePageAllocatorBase {
    static HugePages* hp;
};

} // namespace support

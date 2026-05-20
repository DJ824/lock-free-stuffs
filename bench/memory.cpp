#include "memory.h"

#include <cerrno>
#include <system_error>

#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif

namespace support {

namespace {
size_t const kDefaultPageSize = ::sysconf(_SC_PAGESIZE);
}

HugePages::WarnFn* HugePages::warn_no_1GB_pages = nullptr;
HugePages::WarnFn* HugePages::warn_no_2MB_pages = nullptr;
HugePages* HugePageAllocatorBase::hp = nullptr;

HugePages::HugePages(Type type, size_t total_size) {
    void* p;
    size_t rounded_size;

    for (;;) {
        unsigned flags = 0;
        size_t page_size = kDefaultPageSize;
        bool lock_memory = true;
        if (type != PAGE_DEFAULT) {
            page_size = size_t{1} << type;
            flags = (type << MAP_HUGE_SHIFT) | MAP_HUGETLB;
        } else {
            lock_memory = false;
        }

        rounded_size = (total_size + (page_size - 1)) & ~(page_size - 1);
        unsigned mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | flags;
        if (lock_memory) {
            mmap_flags |= MAP_LOCKED;
        }

        p = ::mmap(nullptr, rounded_size, PROT_READ | PROT_WRITE, mmap_flags, -1, 0);
        if (p != MAP_FAILED) {
            break;
        }

        if (type == PAGE_1GB) {
            type = PAGE_2MB;
            if (warn_no_1GB_pages) {
                warn_no_1GB_pages();
                warn_no_1GB_pages = nullptr;
            }
        } else if (type == PAGE_2MB) {
            type = PAGE_DEFAULT;
            if (warn_no_2MB_pages) {
                warn_no_2MB_pages();
                warn_no_2MB_pages = nullptr;
            }
        } else {
            throw std::system_error(errno, std::system_category(), "mmap");
        }
    }

    beg_ = static_cast<unsigned char*>(p);
    cur_ = beg_;
    end_ = beg_ + rounded_size;
}

HugePages::~HugePages() noexcept {
    if (beg_) {
        ::munmap(beg_, end_ - beg_);
    }
}

HugePages::HugePages(HugePages&& other) noexcept
    : cur_(other.cur_)
    , end_(other.end_)
    , beg_(other.beg_) {
    other.cur_ = nullptr;
    other.end_ = nullptr;
    other.beg_ = nullptr;
}

HugePages& HugePages::operator=(HugePages&& other) noexcept {
    if (this != &other) {
        std::swap(cur_, other.cur_);
        std::swap(end_, other.end_);
        std::swap(beg_, other.beg_);
    }
    return *this;
}

void HugePages::reset() noexcept {
    cur_ = beg_;
}

bool HugePages::empty() const noexcept {
    return cur_ == beg_;
}

size_t HugePages::used() const noexcept {
    return static_cast<size_t>(cur_ - beg_);
}

void* HugePages::allocate(size_t size, std::nothrow_t) noexcept {
    if (static_cast<size_t>(end_ - cur_) < size) {
        return nullptr;
    }
    void* p = cur_;
    cur_ += size;
    return p;
}

void* HugePages::allocate(size_t size) {
    void* p = allocate(size, std::nothrow_t{});
    if (!p) {
        throw std::bad_alloc();
    }
    return p;
}

void HugePages::deallocate(void* p, size_t size) noexcept {
    unsigned char* q = cur_ - size;
    if (q == p) {
        cur_ = q;
    }
}

} // namespace support

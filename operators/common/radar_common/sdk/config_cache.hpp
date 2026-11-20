// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <map>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

namespace holoscan::radar {

inline constexpr std::size_t kDefaultCacheBudgetBytes = 64ULL * 1024 * 1024;

inline std::size_t cache_size_add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::overflow_error("configuration cache byte accounting overflow");
  return a + b;
}

// Dynamic payload; C++ objects and bookkeeping are charged separately.
template <typename V>
std::size_t cache_payload_bytes(const V& value) {
  if constexpr (std::is_arithmetic_v<V>)
    return 0;
  else if constexpr (requires { value.Bytes(); })
    return value.Bytes();
  else
    return value.cache_bytes();
}

template <typename V>
std::size_t cache_payload_bytes(const std::shared_ptr<V>& value) {
  if (!value)
    return 0;
  if constexpr (requires { value->nbytes(); })
    return cache_size_add(sizeof(V) + 64U, value->nbytes());
  else
    return cache_size_add(sizeof(V) + 64U, cache_payload_bytes(*value));
}

// One per operator. Retention accounting includes payloads, objects, key copies
// and a conservative node/control allowance. Builders, current-tick pins, output
// pools, opaque library metadata and MatX-owned caches are outside this budget.
class ConfigCacheBudget {
  struct Key {
    const void* owner;
    std::string text;
    bool operator<(const Key& other) const {
      if (owner != other.owner)
        return std::less<const void*>{}(owner, other.owner);
      return text < other.text;
    }
  };
  struct Entry {
    Key key;
    std::shared_ptr<void> value;
    std::size_t bytes;
  };
  using Entries = std::list<Entry>;

 public:
  explicit ConfigCacheBudget(std::size_t limit = kDefaultCacheBudgetBytes) : limit_(limit) {}
  ConfigCacheBudget(const ConfigCacheBudget&) = delete;
  ConfigCacheBudget& operator=(const ConfigCacheBudget&) = delete;

  // Pins prevent another cache's eviction from invalidating references borrowed
  // earlier in compute(). Only eviction/uncached cleanup waits for GPU work.
  class Tick {
   public:
    Tick(ConfigCacheBudget& budget, std::size_t limit, std::function<void()> barrier)
    : budget_(budget) {
      if (budget_.in_tick_)
        throw std::logic_error("nested configuration-cache tick");
      budget_.barrier_ = std::move(barrier);
      budget_.in_tick_ = true;
      budget_.limit_ = limit;
      budget_.trim(0);
    }
    ~Tick() {
      if (budget_.released_in_tick_)
        budget_.wait();
      budget_.uncached_.reset();
      budget_.pins_.clear();
      budget_.released_in_tick_ = false;
      budget_.in_tick_ = false;
      budget_.barrier_ = {};
    }
    Tick(const Tick&) = delete;
    Tick& operator=(const Tick&) = delete;

   private:
    ConfigCacheBudget& budget_;
  };

  std::size_t bytes() const { return bytes_; }
  std::size_t limit() const { return limit_; }
  std::size_t size(const void* owner) const {
    std::size_t count = 0;
    for (const auto& entry : entries_)
      count += entry.key.owner == owner;
    return count;
  }
  bool contains(const void* owner, const std::string& key) const {
    return index_.find(Key{owner, key}) != index_.end();
  }
  void erase_owner(const void* owner) {
    for (auto it = entries_.begin(); it != entries_.end();) {
      if (it->key.owner == owner) {
        bytes_ -= it->bytes;
        index_.erase(it->key);
        it = entries_.erase(it);
      } else {
        ++it;
      }
    }
  }

  template <typename V, typename Predicate>
  const V* recent(const void* owner, Predicate&& matches) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
      if (it->key.owner != owner)
        continue;
      const auto* value = static_cast<const V*>(it->value.get());
      if (!matches(*value))
        return nullptr;
      pin(it->value);
      entries_.splice(entries_.begin(), entries_, it);
      return value;
    }
    return nullptr;
  }

  template <typename V, typename Builder, typename Sizer>
  const V& get(const void* owner, const std::string& key, const char* what, Builder&& builder,
               Sizer&& sizer, bool& built_before) {
    const auto found = index_.find(Key{owner, key});
    if (found != index_.end()) {
      entries_.splice(entries_.begin(), entries_, found->second);
      pin(found->second->value);
      return *static_cast<const V*>(found->second->value.get());
    }
    // Stable placement: MatX assignment creates a lazy expression, not a rebind.
    auto value = std::make_shared<V>(builder());
    const std::size_t keys = cache_size_add(key.size(), 1U);
    const std::size_t charge = cache_size_add(
        cache_size_add(sizeof(V) + sizeof(Entry) + sizeof(Key) + 160U, sizer(*value)),
        cache_size_add(keys, keys));
    if (built_before && !warned_rebuild_) {
      HOLOSCAN_LOG_WARN(
          "reconfigured to an unseen {} configuration; building it now; "
          "reuse is bounded by cache_budget_bytes",
          what);
      warned_rebuild_ = true;
    }
    built_before = true;
    pin(value);
    if (charge > limit_) {
      if (!warned_oversize_) {
        HOLOSCAN_LOG_WARN(
            "{} needs {} cache bytes, exceeding cache_budget_bytes={}; "
            "using it without caching; raise the budget to retain it",
            what, charge, limit_);
        warned_oversize_ = true;
      }
      released_in_tick_ = true;
      uncached_ = value;
      return *value;
    }
    trim(charge);
    entries_.push_front(Entry{Key{owner, key}, value, charge});
    try {
      index_.emplace(entries_.front().key, entries_.begin());
    } catch (...) {
      entries_.pop_front();
      throw;
    }
    bytes_ += charge;
    return *value;
  }

 private:
  void pin(const std::shared_ptr<void>& value) {
    if (in_tick_)
      pins_.push_back(value);
  }
  void wait() noexcept {
    if (barrier_)
      barrier_();
  }
  void trim(std::size_t incoming) {
    while (!entries_.empty() && (bytes_ > limit_ || incoming > limit_ - bytes_)) {
      if (!warned_capacity_ && incoming != 0) {
        HOLOSCAN_LOG_WARN(
            "cache_budget_bytes={} cannot retain the current working set; "
            "evicting least recently used artifacts; revisits may rebuild",
            limit_);
        warned_capacity_ = true;
      }
      wait();
      auto victim = std::prev(entries_.end());
      bytes_ -= victim->bytes;
      index_.erase(victim->key);
      entries_.erase(victim);
      released_in_tick_ = true;
    }
    uncached_.reset();
  }
  std::size_t limit_;
  std::size_t bytes_{0};
  Entries entries_;
  std::map<Key, Entries::iterator> index_;
  std::vector<std::shared_ptr<void>> pins_;
  std::shared_ptr<void> uncached_;
  std::function<void()> barrier_;
  bool in_tick_{false};
  bool released_in_tick_{false};
  bool warned_rebuild_{false};
  bool warned_oversize_{false};
  bool warned_capacity_{false};
};

template <typename Value>
class ConfigCache {
 public:
  ConfigCache() : budget_(std::make_shared<ConfigCacheBudget>()) {}
  ~ConfigCache() { budget_->erase_owner(this); }
  ConfigCache(const ConfigCache&) = delete;
  ConfigCache& operator=(const ConfigCache&) = delete;
  void bind(const std::shared_ptr<ConfigCacheBudget>& budget) {
    if (budget != budget_) {
      budget_->erase_owner(this);
      budget_ = budget;
    }
  }
  template <typename Builder, typename Sizer>
  const Value& get_or_build(const std::string& key, const char* what, Builder&& builder,
                            Sizer&& sizer) {
    return budget_->get<Value>(this, key, what, std::forward<Builder>(builder),
                               std::forward<Sizer>(sizer), built_before_);
  }
  template <typename Builder>
  const Value& get_or_build(const std::string& key, const char* what, Builder&& builder) {
    return get_or_build(key, what, std::forward<Builder>(builder), [](const Value& value) {
      return cache_payload_bytes(value);
    });
  }
  template <typename Predicate>
  const Value* recent_if(Predicate&& matches) {
    return budget_->recent<Value>(this, std::forward<Predicate>(matches));
  }
  bool contains(const std::string& key) const { return budget_->contains(this, key); }
  std::size_t size() const { return budget_->size(this); }

 private:
  std::shared_ptr<ConfigCacheBudget> budget_;
  bool built_before_{false};
};

}  // namespace holoscan::radar

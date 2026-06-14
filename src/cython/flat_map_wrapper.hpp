#ifndef FLAT_MAP_WRAPPER_HPP
#define FLAT_MAP_WRAPPER_HPP

#if __cplusplus >= 202302L && __has_include(<flat_map>)
  #include <flat_map>
  namespace compat {
      template<class K, class V>
      using flat_map = std::flat_map<K, V>;
  }
#else
  #include <boost/container/flat_map.hpp>
  namespace compat {
      template<class K, class V>
      using flat_map = boost::container::flat_map<K, V>;
  }
#endif

#endif // FLAT_MAP_WRAPPER_HPP

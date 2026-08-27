cmake_minimum_required(VERSION 3.22.1)
project(dpip_wind_host LANGUAGES CXX)

# Nothing links against libmaplibre.so on purpose — MapLibre calls into this
# library, never the other way round, so the only contract is the vtable layout
# transcribed in wind_particle_host.cpp. See the comment at the top of it.
add_library(dpip_wind_host SHARED wind_particle_host.cpp)

target_compile_features(dpip_wind_host PRIVATE cxx_std_17)
target_compile_options(dpip_wind_host PRIVATE -fvisibility=hidden -Wall -Wextra)

find_library(log-lib log)
target_link_libraries(dpip_wind_host ${log-lib})

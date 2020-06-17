macro(pipeline_update_3rdparty SUBMODULE)
    if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
    # Update submodule as needed
        message(STATUS "${SUBMODULE} update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive "${SUBMODULE}"
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout ${SUBMODULE} submodule")
        endif()
    endif()
endmacro()

find_package(Git QUIET)
pipeline_update_3rdparty(cxxopts)

set(CXXOPTS_BUILD_EXAMPLES OFF CACHE INTERNAL "" FORCE)
set(CXXOPTS_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(CXXOPTS_ENABLE_INSTALL OFF CACHE INTERNAL "" FORCE)
set(CXXOPTS_USE_UNICODE_HELP OFF CACHE INTERNAL "" FORCE)

set(CMAKE_FOLDER 3rdparty/cxxopts)
add_subdirectory(cxxopts)
set(CMAKE_FOLDER 3rdparty)

target_link_libraries(MODELS cxxopts)
/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef PATH_DICTIONARY_INTERFACE_H
#define PATH_DICTIONARY_INTERFACE_H

#include "path_dictionary_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /*
    * The path dictionary interface provides an API to share tokens, prim paths and prim lists between systems.
    * Systems that use the same path dictionary can directly share handles to tokens, path and path lists instead of 
    * repeatedly converting internal handles into strings and passing those to another system.
    * Path list handles are unique and immutable to allow caching of associated attribute mappings.
    * All tokens, paths and strings handed out by this system are tied to the lifetime of the path_dictionary_instance_t
    */
    typedef struct path_dictionary_vtable_t
    {
        /* creates tokens from strings */
        ovx_api_result_t (*create_tokens_from_strings)(path_dictionary_context_t* instance,
                                            const ovx_string_t* strings,
                                            size_t num_strings, 
                                            ovx_token_t* out_tokens);

        /* creates paths from arrays of tokens */
        ovx_api_result_t (*create_paths_from_tokens)(path_dictionary_context_t* instance,
                                            const ovx_token_t* tokens_per_path,
                                            const size_t* num_tokens_per_path, 
                                            size_t num_paths, 
                                            ovx_primpath_t* out_prim_paths);

        /* creates paths from strings */
        ovx_api_result_t (*create_paths_from_strings)(path_dictionary_context_t* instance,
                                            const ovx_string_t* path_strings,
                                            size_t num_paths, 
                                            ovx_primpath_t* out_prim_paths);

        /* creates a path list from an array of path */
        ovx_api_result_t (*create_path_list_from_paths)(path_dictionary_context_t* instance,
                                                const ovx_primpath_t* paths,
                                                size_t num_paths, 
                                                ovx_primpath_list_t* out_path_list);

        /* creates a path list from an array of strings */
        ovx_api_result_t (*create_path_list_from_strings)(path_dictionary_context_t* instance,
                                                const ovx_string_t* path_strings,
                                                size_t num_paths,
                                                ovx_primpath_list_t* out_path_list);

        /* destroy a path list */
        ovx_api_result_t (*destroy_path_list)(path_dictionary_context_t* instance,
                                    ovx_primpath_list_t path_list);

        /* Converts an array of tokens into an array of strings. These strings will remain valid while the path_dictionary_interface remains valid */
        ovx_api_result_t (*get_strings_from_tokens)(path_dictionary_context_t* instance,
                                        const ovx_token_t* tokens,
                                        size_t num_tokens,
                                        ovx_string_t* out_strings);

        /*  Converts the list of paths into token arrays using a provided token buffer for storage. 
            Returns the number of paths that fit into the provided token buffer. */
        ovx_api_result_t (*get_tokens_from_paths)(path_dictionary_context_t* instance,
                                        const ovx_primpath_t* prim_paths, /* array of prim paths to obtain the tokens for */
                                        size_t num_paths,                /* number of prim paths in array */
                                        ovx_token_t* token_buffer,        /* token storage buffer */
                                        size_t token_buffer_size,        /* size of the token storage buffer */
                                        ovx_token_t** out_tokens_per_path, /* array of pointer to token arrays of size num_paths. This will point at the start of a token array for each path in prim_paths*/
                                        size_t* out_num_tokens_per_path,  /* array of sizes of size num_paths that holds the number of tokens per prim path */
                                        size_t* out_num_paths_processed); /* output containing the number of prim paths that could be fit into the token buffer */


        /* Retrieve the number of paths from a path list */
        ovx_api_result_t (*get_num_paths_from_path_list)(path_dictionary_context_t* instance,
                                            ovx_primpath_list_t path_list,
                                            size_t* out_num_paths);

        /*  Retrieve the paths represented by a path list. 
            Returns true if the remaining paths from the given start_offset fit into max_paths */
        ovx_api_result_t (*get_paths_from_path_list)(path_dictionary_context_t* instance,
                                            ovx_primpath_list_t path_list,
                                            size_t start_offset,
                                            size_t max_paths,
                                            ovx_primpath_t* out_paths,
                                            size_t* out_num_paths);

        /*release resources associated with a returned error string*/
        void (*release_error)(path_dictionary_context_t* context, 
                                ovx_string_t error);

    } path_dictionary_vtable_t;

#ifdef __cplusplus
}
#endif

#include "path_dictionary_utils.h"

#endif /* PATH_DICTIONARY_INTERFACE_H */

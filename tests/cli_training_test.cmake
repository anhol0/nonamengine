# Each invocation owns a unique directory and removes it before reporting errors.
string(RANDOM LENGTH 24 ALPHABET 0123456789abcdef suffix)
set(directory "${BINARY_DIR}/cli training ${suffix}")
file(MAKE_DIRECTORY "${directory}")
set(failure "")

execute_process(COMMAND "${PROGRAM}" train --iterations 2 --batch-size 8
    --dir "${directory}" --output trained
    RESULT_VARIABLE status OUTPUT_VARIABLE trained_output ERROR_VARIABLE error
    TIMEOUT 30)
if(NOT "${status}" STREQUAL "0")
    set(failure "Training failed: ${status}\n${error}")
elseif(NOT EXISTS "${directory}/trained.json" OR NOT EXISTS "${directory}/trained.bin")
    set(failure "Training did not export both model files")
else()
    file(SIZE "${directory}/trained.bin" size)
    if(NOT size EQUAL 516)
        set(failure "Incorrect trained model binary size: ${size}")
    endif()
    execute_process(COMMAND "${PROGRAM}" load "${directory}/trained.json"
        WORKING_DIRECTORY "${SOURCE_DIR}"
        RESULT_VARIABLE status OUTPUT_VARIABLE loaded_output ERROR_VARIABLE error
        TIMEOUT 30)
    if(NOT "${status}" STREQUAL "0")
        set(failure "Reload failed: ${status}\n${error}")
    else()
        string(FIND "${trained_output}" "Validation results:" position)
        if(position LESS 0)
            set(failure "Training did not print predictions")
        else()
            string(SUBSTRING "${trained_output}" ${position} -1 expected)
            string(STRIP "${loaded_output}" actual)
            string(STRIP "${expected}" expected)
            if(NOT actual STREQUAL expected)
                set(failure "Reload changed validation predictions or metrics")
            endif()
        endif()
    endif()
endif()

if(failure STREQUAL "")
    # Export failure must propagate to the process exit status after training.
    file(WRITE "${directory}/blocked" "This is a file, not a directory.")
    execute_process(COMMAND "${PROGRAM}" train --iterations 1 --batch-size 4
        --dir "${directory}/blocked" --output trained
        RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error
        TIMEOUT 30)
    if(NOT status MATCHES "^[0-9]+$" OR status EQUAL 0 OR error STREQUAL "")
        set(failure "Export failure was not reported: ${status}\n${error}")
    endif()
endif()

file(REMOVE_RECURSE "${directory}")
if(NOT failure STREQUAL "")
    message(FATAL_ERROR "${failure}")
endif()

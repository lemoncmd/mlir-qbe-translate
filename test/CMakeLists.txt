configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(QBE_TEST_DEPENDS
        FileCheck count not
        qbe-opt
        qbe-translate
        )

add_lit_testsuite(check-qbe "Running the QBE regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${QBE_TEST_DEPENDS}
        )
set_target_properties(check-qbe PROPERTIES FOLDER "Tests")

add_lit_testsuites(QBE ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${QBE_TEST_DEPENDS})

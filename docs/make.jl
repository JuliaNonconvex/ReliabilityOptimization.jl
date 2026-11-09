using DocumenterFragments: build_fragment

# The Ipopt C library prints a one-time banner to stdout on load; filter it
# out of the doctest outputs.
build_fragment(
    @__DIR__;
    checkdocs=:public,
    doctestfilters=[
        r"This program contains Ipopt",
        r"Ipopt is released as open source code",
        r"For more information visit",
        r"^ *\*+ *$",
    ],
)

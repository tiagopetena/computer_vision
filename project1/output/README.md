output: a directory where your application should produce all the generated files (otherwise stated in the
problem). This folder and all its contents must be added to the artifacts path on the .gitlab-ci.yml
setup.

In case of a problem that asks for producing images automatically, use a convention for naming the im-
ages that easily identifies them. For example, you can use [io]-<question #>-<part>-<counter>.png,

where i will be used for input and o for output images, <part> may be omitted if the question does not have
one, and that <counter> must be auto-incremented starting from 0. Any other configurations are possible
as long as they are consistent.
# Auto initialization for in-place operations

## Status

PROPOSED

Discussed by: Paul Dubs, Alex Black on 25. November 2019


## Context

Some operations work in-place on the inputs that they are given in ND4J, but in SameDiff the same operations will
generate an array from a given shape. Examples for this include `BernoulliDistribution`, and other random ops, that
effectively initialize the array that they are given.

From a consistency point of view, it would be nice if both API's would support both ways of using those ops.


## Decision

We introduce an option to mark inputs as `inPlace = true` to make it clear that this input is going to be changed
in-place. In addition we introduce an option `supportsInPlaceInit = true` to mark an input as initialize-able. If the
`supportsInPlaceInit` option is enabled, two signatures for the Op will be created, one that takes an input, and one
that takes the appropriate shape and data type information in its stead.  


## Consequences

### Advantages
* We get support for both in-place and initialization use-cases
* Support for this use-case is explicitly defined in the DSL instead of implicit support by the code generator

### Disadvantages
* Codegen becomes more complex, as it requires us to generate more signatures
# Global Enums

## Status
Proposed

## Context
As we moved from generating just the ND4J namespaces to also generating SameDiff namespaces it has become apparent, that we often want to reuse Enums across namespaces and not just within ops of the same namespace. 

# Proposal
We lift Enum definitions from being just a special case in Arg definitions to be an actual first class concept, which can be defined at the top-level.

This also opens up the possibility to have detailed documentation for each option of an enum. 

In order to keep complexity down, we explicitly do not support parameterized enums.

### Example
```kotlin
val Foo = Enum("Foo"){
    javaPackage = "foo.bar.baz"
    Option("EGGS"){ "Single line Docstring"}
    Option("SPAM"){ """
        Multi
        Line
        Docstring
    """}
}

// Within some op:
Arg(ENUM, "foobar"){ enum = Foo; defaultValue = Foo.option("EGGS")}
```
  
## Consequences
### Advantages
* Support for reusable enums
* Better documentation for enum options
* Compile-time enum reference checks
* Construct-time default value reference checks
* Allows us to remove all workarounds that were used to reference pre-existing enums 

### Disadvantages
* Yet another top-level concept that needs to be explained
* Only a "limited" enum
* Two ways of creating an enum if we keep the original way from [ADR 0006 "Op Specific Enums"](0006-op_specific_enums.md) around

## Open Questions
* Should we keep the original way of doing enums? (see [ADR 0006 "Op Specific Enums"](0006-op_specific_enums.md))
    * **Pro**
      * No need to change anything that already exists
    * **Con**
      * Potential for confusion
      * two points to remember during codegen
      * one way is likely to be favored by the devs, making the other more likely to become unmaintained over time
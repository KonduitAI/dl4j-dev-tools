# Configuration Objects

## Status

Proposal


## Context
Some Ops (esp. convolution) have many parameters. Many of them can have reasonable defaults, but even then creating
signatures for evey reasonable configuration may be impossible, as those signatures would require different naming in
order to be actually distinguishable from each other.

In other cases, an op may have a lot of same typed parameters that are required (e.g. GRU, LSTM, SRU) but it is very
easy to mix them up. 

For both of those cases (many optional parameters, easily mixed up required parameters) it is reasonable to use a
config holder with builder pattern in languages that do not support named or default parameters. In languages that
support named and default parameters, we will likely not have to use this feature. 

## Proposal

We introduce a `Config("name", a, b, c)` section within the op context. It will specify a name for the configuration
class, and which parameters (i.e. inputs and args) can be configured through it.
 
We extend the `Signature` feature introduced in ADR 0005 "Optional Parameters and Signatures" to support configurations.

At construction time, `Signature` will check that with all given parameters, it is actually possible to invoke the op,
given the previously defined inputs and args.

### Example
The following shows a simple example, with not too many actual inputs. It is meant to demonstrate how it would look like
to use the config section.

```kotlin
Op("SRUCell"){
  val x = Input(NUMERIC, "x"){}
  val cLast = Input(NUMERIC, "cLast"){}
  val weights = Input(NUMERIC, "weights"){}
  val bias = Input(NUMERIC, "bias"){}
  
  val config = Config("SRUWeights", weights, bias)

  Output(NUMERIC, "out"){}

  Signature(x, cLast, config)
}
```
 
## Consequences

### Advantages
* We make it possible to define a more convenient op interface for languages where default and named arguments are not 
  available
* Config holder classes are defined within their ops instead of being part of the runtime environment

### Disadvantages
* Reuse of config holders across classes is not supported
* This is yet another feature that aims at languages that do not support default and named parameters

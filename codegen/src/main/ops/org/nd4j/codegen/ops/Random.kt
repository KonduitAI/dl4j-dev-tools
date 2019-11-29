package org.nd4j.codegen.ops

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*

fun Random() = Namespace("Random") {


    var legacyRandom = Op("legacyRandom"){
        isAbstract = true
        legacy = true
        javaPackage = "org.nd4j.linalg.api.ops.random.impl"
        Arg(DATA_TYPE, "datatype"){ description = "Data type of the output variable"}
        Arg(LONG, "shape") { count = AtLeast(0); description = "Shape of the new random %INPUT_TYPE%, as a 1D array" }
        Output(NUMERIC, "output") { description = "Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution" }
    }


    Op("bernoulli", legacyRandom) {
        javaOpClass = "BernoulliDistribution"
        val p = Arg(FLOATING_POINT, "p") { description = "Probability of value 1" }

        Signature(p, args.get(0), args.get(1))      //probability, datatype, shape

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a Bernoulli distribution,
            with the specified probability. Array values will have value 1 with probability P and value 0 with probability
            1-P.
            """.trimIndent()
        }
    }

    Op("binomial", legacyRandom) {
        javaOpClass = "BinomialDistribution"

        val n = Arg(INT, "nTrials") { description = "Number of trials parameter for the binomial distribution" }
        val p = Arg(FLOATING_POINT, "p") { description = "Probability of success for each trial" }

        Signature(n, p, args.get(0), args.get(1))      //trials, probability, datatype, shape

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a Binomial distribution,
            with the specified number of trials and probability.
            """.trimIndent()
        }
    }

    Op("exponential") {
        javaPackage = "org.nd4j.linalg.api.ops.random.custom"
        javaOpClass = "RandomExponential"

        val lambda = Arg(FLOATING_POINT, "lambda") { description = "lambda parameter" }
        Constraint("Must be positive") { lambda gt 0 }
        Arg(DATA_TYPE, "datatype"){ description = "Data type of the output variable"}
        Arg(LONG, "shape") { count = AtLeast(0); description = "Shape of the new random %INPUT_TYPE%, as a 1D array" }

        AllParamSignature()


        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a exponential distribution:
            P(x) = lambda * exp(-lambda * x)
            """.trimIndent()
        }
    }

    Op("logNormal", legacyRandom) {
        javaOpClass = "LogNormalDistribution"

        val m = Arg(FLOATING_POINT, "mean") { description = "Mean value for the random array" }
        val s = Arg(FLOATING_POINT, "stddev") { description = "Standard deviation for the random array" }

        Signature(m, s, args.get(0), args.get(1))      //mean, stddev, datatype, shape

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a Log Normal distribution,
            i.e., {@code log(x) ~ N(mean, stdev)}
            """.trimIndent()
        }
    }

    Op("normal", legacyRandom) {
        javaPackage = "org.nd4j.linalg.api.ops.random.impl"
        javaOpClass = "GaussianDistribution"

        val m = Arg(FLOATING_POINT, "mean") { description = "Mean value for the random array" }
        val s = Arg(FLOATING_POINT, "stddev") { description = "Standard deviation for the random array" }

        Signature(m, s, args.get(0), args.get(1))      //mean, stddev, datatype, shape

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a Gaussian (normal) distribution,
            N(mean, stdev)<br>
            """.trimIndent()
        }
    }

    Op("normalTruncated", legacyRandom) {
        javaOpClass = "TruncatedNormalDistribution"
        val m = Arg(FLOATING_POINT, "mean") { description = "Mean value for the random array" }
        val s = Arg(FLOATING_POINT, "stddev") { description = "Standard deviation for the random array" }

        Signature(m, s, args.get(0), args.get(1))      //mean, stddev, datatype, shape

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a Gaussian (normal) distribution,
            N(mean, stdev). However, any values more than 1 standard deviation from the mean are dropped and re-sampled
            """.trimIndent()
        }
    }

    Op("uniform", legacyRandom) {
        javaOpClass = "UniformDistribution"

        val min = Arg(FLOATING_POINT, "min") { description = "Minimum value" }
        val max = Arg(FLOATING_POINT, "max") { description = "Maximum value." }

        Signature(min, max, args.get(0), args.get(1))      //min, max, datatype, shape

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a uniform distribution,
            U(min,max)
            """.trimIndent()
        }
    }
}
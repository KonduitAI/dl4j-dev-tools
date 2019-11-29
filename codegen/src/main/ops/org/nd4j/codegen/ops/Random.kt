package org.nd4j.codegen.ops

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*

fun Random() = Namespace("Random") {


    var legacyRandom = Op("legacyRandom"){
        javaPackage = "org.nd4j.linalg.api.ops.random.impl"
        Arg(DATA_TYPE, "datatype"){ description = "Data type of the output variable"}
        Arg(INT, "shape") { count = AtLeast(0); description = "Shape of the new random %INPUT_TYPE%, as a 1D array" }
        Output(NUMERIC, "output") { description = "Tensor with the given shape where values are randomly sampled according to a %OP_NAME% distribution" }
    }


    Op("bernoulli", legacyRandom) {
        javaOpClass = "BernoulliDistribution"

        Arg(FLOATING_POINT, "p") { description = "Probability of value 1" }


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

        Arg(INT, "nTrials") { description = "Number of trials parameter for the binomial distribution" }
        Arg(FLOATING_POINT, "p") { description = "Probability of success for each trial" }

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


        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a exponential distribution:
            P(x) = lambda * exp(-lambda * x)
            """.trimIndent()
        }
    }

    Op("logNormal", legacyRandom) {
        javaOpClass = "LogNormalDistribution"

        Arg(FLOATING_POINT, "mean") { description = "Mean value for the random array" }
        Arg(FLOATING_POINT, "stddev") { description = "Standard deviation for the random array" }

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

        Arg(FLOATING_POINT, "mean") { description = "Mean value for the random array" }
        Arg(FLOATING_POINT, "stddev") { description = "Standard deviation for the random array" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a Gaussian (normal) distribution,
            N(mean, stdev)<br>
            """.trimIndent()
        }
    }

    Op("normalTruncated", legacyRandom) {
        javaOpClass = "TruncatedNormalDistribution"
        Arg(FLOATING_POINT, "mean") { description = "Mean value for the random array" }
        Arg(FLOATING_POINT, "stddev") { description = "Standard deviation for the random array" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a Gaussian (normal) distribution,
            N(mean, stdev). However, any values more than 1 standard deviation from the mean are dropped and re-sampled
            """.trimIndent()
        }
    }

    Op("uniform", legacyRandom) {
        javaOpClass = "UniformDistribution"

        Arg(FLOATING_POINT, "min") { description = "Minimum value" }
        Arg(FLOATING_POINT, "max") { description = "Maximum value." }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Generate a new random %INPUT_TYPE%, where values are randomly sampled according to a uniform distribution,
            U(min,max)
            """.trimIndent()
        }
    }
}
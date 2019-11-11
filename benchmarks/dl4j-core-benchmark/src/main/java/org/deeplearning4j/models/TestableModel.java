package org.deeplearning4j.models;

import org.deeplearning4j.nn.api.Model;

/**
 * Generic interface for testable models.
 */
public interface TestableModel {

    public Model init();

    public ModelMetaData metaData();
}

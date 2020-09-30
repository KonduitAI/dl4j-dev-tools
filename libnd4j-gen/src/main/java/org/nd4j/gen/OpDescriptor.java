package org.nd4j.gen;

import lombok.Builder;
import lombok.Data;
import org.nd4j.common.base.Preconditions;

import java.io.Serializable;
import java.util.List;

@Data
@Builder(toBuilder = true)
public class OpDescriptor implements Serializable  {
    private String name;
    private int nIn,nOut,tArgs,iArgs;
    private boolean inplaceAble;
    private List<String> inArgNames;
    private List<String> outArgNames;
    private List<String> tArgNames;
    private List<String> iArgNames;
    private List<String> bArgNames;

    public void validate() {
        if(nIn >= 0 && nIn != inArgNames.size()) {
            System.err.println("In arg names was not equal to number of inputs found for op " + name);
        }

        if(nOut >= 0 && nOut != outArgNames.size()) {
            System.err.println("Output arg names was not equal to number of outputs found for op " + name);
        }

        if(tArgs >= 0 && tArgs != tArgNames.size()) {
            System.err.println("T arg names was not equal to number of T found for op " + name);
        }
        if(iArgs >= 0 && iArgs != iArgNames.size()) {
            System.err.println("Integer arg names was not equal to number of integer args found for op " + name);
        }
       /* if(nIn >= 0)
            Preconditions.checkState(nIn == inArgNames.size(),"In arg names was not equal to number of inputs found for op " + name);
        if(nOut >= 0)
            Preconditions.checkState(nOut == outArgNames.size(),"Output arg names was not equal to number of outputs found for op " + name);
        if(tArgs >= 0)
            Preconditions.checkState(tArgs == tArgNames.size(),"T arg names was not equal to number of T found for op " + name);
        if(iArgs >= 0)
            Preconditions.checkState(iArgs == iArgNames.size(),"Integer arg names was not equal to number of integer args found for op " + name);
  */  }

}

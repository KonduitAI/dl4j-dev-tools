from onnx.defs import get_all_schemas
from onnx import NodeProto
from google.protobuf import text_format


nodes = []
schemas = get_all_schemas()


def load_node(input_str):
    """
    Return a node
    :param input_str:
    :return:
    """
    node_proto = NodeProto()
    text_format.Parse(input_str,node_proto)
    return node_proto

def create_node_from_schema(schema):

    """
    Convert an OpSchema to a NodeProto
    :param schema:  the input OpSchema
    :return: the equivalent NodeProto
    """

    node_proto = NodeProto()
    for attribute in schema.attributes:
        attr_value = schema.attributes[attribute]
        node_proto.attribute.append(attr_value.default_value)
    node_proto.op_type = schema.name
    node_proto.doc_string = schema.doc
    node_proto.name = schema.name
    for input_arr in schema.inputs:
        if node_proto.input is None:
            node_proto.input = []
        node_proto.input.append(input_arr.name)
    for output_arr in schema.outputs:
        if node_proto.output is None:
            node_proto.output = []
        node_proto.output.append(output_arr.name)
    return node_proto


nodes = [create_node_from_schema(schema) for schema
         in sorted(schemas, key=lambda s: s.name)]

with open('onnx.pbtxt', 'w+') as f:
    for node in nodes:
        message_to_string = text_format.MessageToString(node, as_utf8=True)
        node_2 = load_node(message_to_string)
        f.write(message_to_string + '--\n')

with open('onnx.pbtxt','r') as f:
    nodes = [load_node(node_str) for node_str in f.read().split('--\n')]
    print(nodes)

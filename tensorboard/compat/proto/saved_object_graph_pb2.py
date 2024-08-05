# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorboard/compat/proto/saved_object_graph.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2
from tensorboard.compat.proto import tensor_shape_pb2 as tensorboard_dot_compat_dot_proto_dot_tensor__shape__pb2
from tensorboard.compat.proto import types_pb2 as tensorboard_dot_compat_dot_proto_dot_types__pb2
from tensorboard.compat.proto import variable_pb2 as tensorboard_dot_compat_dot_proto_dot_variable__pb2
from tensorboard.compat.proto import versions_pb2 as tensorboard_dot_compat_dot_proto_dot_versions__pb2
from tensorboard.compat.proto import struct_pb2 as tensorboard_dot_compat_dot_proto_dot_struct__pb2
from tensorboard.compat.proto import trackable_object_graph_pb2 as tensorboard_dot_compat_dot_proto_dot_trackable__object__graph__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n1tensorboard/compat/proto/saved_object_graph.proto\x12\x0btensorboard\x1a\x19google/protobuf/any.proto\x1a+tensorboard/compat/proto/tensor_shape.proto\x1a$tensorboard/compat/proto/types.proto\x1a\'tensorboard/compat/proto/variable.proto\x1a\'tensorboard/compat/proto/versions.proto\x1a%tensorboard/compat/proto/struct.proto\x1a\x35tensorboard/compat/proto/trackable_object_graph.proto\"\xeb\x01\n\x10SavedObjectGraph\x12\'\n\x05nodes\x18\x01 \x03(\x0b\x32\x18.tensorboard.SavedObject\x12P\n\x12\x63oncrete_functions\x18\x02 \x03(\x0b\x32\x34.tensorboard.SavedObjectGraph.ConcreteFunctionsEntry\x1a\\\n\x16\x43oncreteFunctionsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x31\n\x05value\x18\x02 \x01(\x0b\x32\".tensorboard.SavedConcreteFunction:\x02\x38\x01\"\xdd\x07\n\x0bSavedObject\x12S\n\x08\x63hildren\x18\x01 \x03(\x0b\x32\x41.tensorboard.TrackableObjectGraph.TrackableObject.ObjectReference\x12W\n\x0c\x64\x65pendencies\x18\x0f \x03(\x0b\x32\x41.tensorboard.TrackableObjectGraph.TrackableObject.ObjectReference\x12_\n\x0eslot_variables\x18\x03 \x03(\x0b\x32G.tensorboard.TrackableObjectGraph.TrackableObject.SlotVariableReference\x12\x33\n\x0buser_object\x18\x04 \x01(\x0b\x32\x1c.tensorboard.SavedUserObjectH\x00\x12(\n\x05\x61sset\x18\x05 \x01(\x0b\x32\x17.tensorboard.SavedAssetH\x00\x12.\n\x08\x66unction\x18\x06 \x01(\x0b\x32\x1a.tensorboard.SavedFunctionH\x00\x12.\n\x08variable\x18\x07 \x01(\x0b\x32\x1a.tensorboard.SavedVariableH\x00\x12H\n\x16\x62\x61re_concrete_function\x18\x08 \x01(\x0b\x32&.tensorboard.SavedBareConcreteFunctionH\x00\x12.\n\x08\x63onstant\x18\t \x01(\x0b\x32\x1a.tensorboard.SavedConstantH\x00\x12.\n\x08resource\x18\n \x01(\x0b\x32\x1a.tensorboard.SavedResourceH\x00\x12\x36\n\x0f\x63\x61ptured_tensor\x18\x0c \x01(\x0b\x32\x1b.tensorboard.CapturedTensorH\x00\x12G\n\x10saveable_objects\x18\x0b \x03(\x0b\x32-.tensorboard.SavedObject.SaveableObjectsEntry\x12\x17\n\x0fregistered_name\x18\r \x01(\t\x12\x33\n\x15serialized_user_proto\x18\x0e \x01(\x0b\x32\x14.google.protobuf.Any\x12\x18\n\x10registered_saver\x18\x10 \x01(\t\x1aS\n\x14SaveableObjectsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12*\n\x05value\x18\x02 \x01(\x0b\x32\x1b.tensorboard.SaveableObject:\x02\x38\x01\x42\x06\n\x04kindJ\x04\x08\x02\x10\x03R\nattributes\"e\n\x0fSavedUserObject\x12\x12\n\nidentifier\x18\x01 \x01(\t\x12(\n\x07version\x18\x02 \x01(\x0b\x32\x17.tensorboard.VersionDef\x12\x14\n\x08metadata\x18\x03 \x01(\tB\x02\x18\x01\"*\n\nSavedAsset\x12\x1c\n\x14\x61sset_file_def_index\x18\x01 \x01(\x05\"]\n\rSavedFunction\x12\x1a\n\x12\x63oncrete_functions\x18\x01 \x03(\t\x12\x30\n\rfunction_spec\x18\x02 \x01(\x0b\x32\x19.tensorboard.FunctionSpec\"9\n\x0e\x43\x61pturedTensor\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x19\n\x11\x63oncrete_function\x18\x02 \x01(\t\"\xaa\x01\n\x15SavedConcreteFunction\x12\x14\n\x0c\x62ound_inputs\x18\x02 \x03(\x05\x12\x43\n\x1d\x63\x61nonicalized_input_signature\x18\x03 \x01(\x0b\x32\x1c.tensorboard.StructuredValue\x12\x36\n\x10output_signature\x18\x04 \x01(\x0b\x32\x1c.tensorboard.StructuredValue\"\xae\x01\n\x19SavedBareConcreteFunction\x12\x1e\n\x16\x63oncrete_function_name\x18\x01 \x01(\t\x12\x19\n\x11\x61rgument_keywords\x18\x02 \x03(\t\x12$\n\x1c\x61llowed_positional_arguments\x18\x03 \x01(\x03\x12\x30\n\rfunction_spec\x18\x04 \x01(\x0b\x32\x19.tensorboard.FunctionSpec\"\"\n\rSavedConstant\x12\x11\n\toperation\x18\x01 \x01(\t\"\xdc\x02\n\rSavedVariable\x12$\n\x05\x64type\x18\x01 \x01(\x0e\x32\x15.tensorboard.DataType\x12,\n\x05shape\x18\x02 \x01(\x0b\x32\x1d.tensorboard.TensorShapeProto\x12\x11\n\ttrainable\x18\x03 \x01(\x08\x12=\n\x0fsynchronization\x18\x04 \x01(\x0e\x32$.tensorboard.VariableSynchronization\x12\x35\n\x0b\x61ggregation\x18\x05 \x01(\x0e\x32 .tensorboard.VariableAggregation\x12\x0c\n\x04name\x18\x06 \x01(\t\x12\x0e\n\x06\x64\x65vice\x18\x07 \x01(\t\x12P\n,experimental_distributed_variable_components\x18\x08 \x03(\x0b\x32\x1a.tensorboard.SavedVariable\"\xfe\x01\n\x0c\x46unctionSpec\x12\x31\n\x0b\x66ullargspec\x18\x01 \x01(\x0b\x32\x1c.tensorboard.StructuredValue\x12\x11\n\tis_method\x18\x02 \x01(\x08\x12\x35\n\x0finput_signature\x18\x05 \x01(\x0b\x32\x1c.tensorboard.StructuredValue\x12\x39\n\x0bjit_compile\x18\x06 \x01(\x0e\x32$.tensorboard.FunctionSpec.JitCompile\"*\n\nJitCompile\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\x06\n\x02ON\x10\x01\x12\x07\n\x03OFF\x10\x02J\x04\x08\x03\x10\x04J\x04\x08\x04\x10\x05\"\x1f\n\rSavedResource\x12\x0e\n\x06\x64\x65vice\x18\x01 \x01(\t\"A\n\x0eSaveableObject\x12\x15\n\rsave_function\x18\x02 \x01(\x05\x12\x18\n\x10restore_function\x18\x03 \x01(\x05\x42ZZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto\xf8\x01\x01\x62\x06proto3')



_SAVEDOBJECTGRAPH = DESCRIPTOR.message_types_by_name['SavedObjectGraph']
_SAVEDOBJECTGRAPH_CONCRETEFUNCTIONSENTRY = _SAVEDOBJECTGRAPH.nested_types_by_name['ConcreteFunctionsEntry']
_SAVEDOBJECT = DESCRIPTOR.message_types_by_name['SavedObject']
_SAVEDOBJECT_SAVEABLEOBJECTSENTRY = _SAVEDOBJECT.nested_types_by_name['SaveableObjectsEntry']
_SAVEDUSEROBJECT = DESCRIPTOR.message_types_by_name['SavedUserObject']
_SAVEDASSET = DESCRIPTOR.message_types_by_name['SavedAsset']
_SAVEDFUNCTION = DESCRIPTOR.message_types_by_name['SavedFunction']
_CAPTUREDTENSOR = DESCRIPTOR.message_types_by_name['CapturedTensor']
_SAVEDCONCRETEFUNCTION = DESCRIPTOR.message_types_by_name['SavedConcreteFunction']
_SAVEDBARECONCRETEFUNCTION = DESCRIPTOR.message_types_by_name['SavedBareConcreteFunction']
_SAVEDCONSTANT = DESCRIPTOR.message_types_by_name['SavedConstant']
_SAVEDVARIABLE = DESCRIPTOR.message_types_by_name['SavedVariable']
_FUNCTIONSPEC = DESCRIPTOR.message_types_by_name['FunctionSpec']
_SAVEDRESOURCE = DESCRIPTOR.message_types_by_name['SavedResource']
_SAVEABLEOBJECT = DESCRIPTOR.message_types_by_name['SaveableObject']
_FUNCTIONSPEC_JITCOMPILE = _FUNCTIONSPEC.enum_types_by_name['JitCompile']
SavedObjectGraph = _reflection.GeneratedProtocolMessageType('SavedObjectGraph', (_message.Message,), {

  'ConcreteFunctionsEntry' : _reflection.GeneratedProtocolMessageType('ConcreteFunctionsEntry', (_message.Message,), {
    'DESCRIPTOR' : _SAVEDOBJECTGRAPH_CONCRETEFUNCTIONSENTRY,
    '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
    # @@protoc_insertion_point(class_scope:tensorboard.SavedObjectGraph.ConcreteFunctionsEntry)
    })
  ,
  'DESCRIPTOR' : _SAVEDOBJECTGRAPH,
  '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SavedObjectGraph)
  })
_sym_db.RegisterMessage(SavedObjectGraph)
_sym_db.RegisterMessage(SavedObjectGraph.ConcreteFunctionsEntry)

SavedObject = _reflection.GeneratedProtocolMessageType('SavedObject', (_message.Message,), {

  'SaveableObjectsEntry' : _reflection.GeneratedProtocolMessageType('SaveableObjectsEntry', (_message.Message,), {
    'DESCRIPTOR' : _SAVEDOBJECT_SAVEABLEOBJECTSENTRY,
    '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
    # @@protoc_insertion_point(class_scope:tensorboard.SavedObject.SaveableObjectsEntry)
    })
  ,
  'DESCRIPTOR' : _SAVEDOBJECT,
  '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SavedObject)
  })
_sym_db.RegisterMessage(SavedObject)
_sym_db.RegisterMessage(SavedObject.SaveableObjectsEntry)

SavedUserObject = _reflection.GeneratedProtocolMessageType('SavedUserObject', (_message.Message,), {
  'DESCRIPTOR' : _SAVEDUSEROBJECT,
  '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SavedUserObject)
  })
_sym_db.RegisterMessage(SavedUserObject)

SavedAsset = _reflection.GeneratedProtocolMessageType('SavedAsset', (_message.Message,), {
  'DESCRIPTOR' : _SAVEDASSET,
  '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SavedAsset)
  })
_sym_db.RegisterMessage(SavedAsset)

SavedFunction = _reflection.GeneratedProtocolMessageType('SavedFunction', (_message.Message,), {
  'DESCRIPTOR' : _SAVEDFUNCTION,
  '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SavedFunction)
  })
_sym_db.RegisterMessage(SavedFunction)

CapturedTensor = _reflection.GeneratedProtocolMessageType('CapturedTensor', (_message.Message,), {
  'DESCRIPTOR' : _CAPTUREDTENSOR,
  '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.CapturedTensor)
  })
_sym_db.RegisterMessage(CapturedTensor)

SavedConcreteFunction = _reflection.GeneratedProtocolMessageType('SavedConcreteFunction', (_message.Message,), {
  'DESCRIPTOR' : _SAVEDCONCRETEFUNCTION,
  '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SavedConcreteFunction)
  })
_sym_db.RegisterMessage(SavedConcreteFunction)

SavedBareConcreteFunction = _reflection.GeneratedProtocolMessageType('SavedBareConcreteFunction', (_message.Message,), {
  'DESCRIPTOR' : _SAVEDBARECONCRETEFUNCTION,
  '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SavedBareConcreteFunction)
  })
_sym_db.RegisterMessage(SavedBareConcreteFunction)

SavedConstant = _reflection.GeneratedProtocolMessageType('SavedConstant', (_message.Message,), {
  'DESCRIPTOR' : _SAVEDCONSTANT,
  '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SavedConstant)
  })
_sym_db.RegisterMessage(SavedConstant)

SavedVariable = _reflection.GeneratedProtocolMessageType('SavedVariable', (_message.Message,), {
  'DESCRIPTOR' : _SAVEDVARIABLE,
  '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SavedVariable)
  })
_sym_db.RegisterMessage(SavedVariable)

FunctionSpec = _reflection.GeneratedProtocolMessageType('FunctionSpec', (_message.Message,), {
  'DESCRIPTOR' : _FUNCTIONSPEC,
  '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.FunctionSpec)
  })
_sym_db.RegisterMessage(FunctionSpec)

SavedResource = _reflection.GeneratedProtocolMessageType('SavedResource', (_message.Message,), {
  'DESCRIPTOR' : _SAVEDRESOURCE,
  '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SavedResource)
  })
_sym_db.RegisterMessage(SavedResource)

SaveableObject = _reflection.GeneratedProtocolMessageType('SaveableObject', (_message.Message,), {
  'DESCRIPTOR' : _SAVEABLEOBJECT,
  '__module__' : 'tensorboard.compat.proto.saved_object_graph_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.SaveableObject)
  })
_sym_db.RegisterMessage(SaveableObject)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'ZUgithub.com/tensorflow/tensorflow/tensorflow/go/core/protobuf/for_core_protos_go_proto\370\001\001'
  _SAVEDOBJECTGRAPH_CONCRETEFUNCTIONSENTRY._options = None
  _SAVEDOBJECTGRAPH_CONCRETEFUNCTIONSENTRY._serialized_options = b'8\001'
  _SAVEDOBJECT_SAVEABLEOBJECTSENTRY._options = None
  _SAVEDOBJECT_SAVEABLEOBJECTSENTRY._serialized_options = b'8\001'
  _SAVEDUSEROBJECT.fields_by_name['metadata']._options = None
  _SAVEDUSEROBJECT.fields_by_name['metadata']._serialized_options = b'\030\001'
  _SAVEDOBJECTGRAPH._serialized_start=353
  _SAVEDOBJECTGRAPH._serialized_end=588
  _SAVEDOBJECTGRAPH_CONCRETEFUNCTIONSENTRY._serialized_start=496
  _SAVEDOBJECTGRAPH_CONCRETEFUNCTIONSENTRY._serialized_end=588
  _SAVEDOBJECT._serialized_start=591
  _SAVEDOBJECT._serialized_end=1580
  _SAVEDOBJECT_SAVEABLEOBJECTSENTRY._serialized_start=1471
  _SAVEDOBJECT_SAVEABLEOBJECTSENTRY._serialized_end=1554
  _SAVEDUSEROBJECT._serialized_start=1582
  _SAVEDUSEROBJECT._serialized_end=1683
  _SAVEDASSET._serialized_start=1685
  _SAVEDASSET._serialized_end=1727
  _SAVEDFUNCTION._serialized_start=1729
  _SAVEDFUNCTION._serialized_end=1822
  _CAPTUREDTENSOR._serialized_start=1824
  _CAPTUREDTENSOR._serialized_end=1881
  _SAVEDCONCRETEFUNCTION._serialized_start=1884
  _SAVEDCONCRETEFUNCTION._serialized_end=2054
  _SAVEDBARECONCRETEFUNCTION._serialized_start=2057
  _SAVEDBARECONCRETEFUNCTION._serialized_end=2231
  _SAVEDCONSTANT._serialized_start=2233
  _SAVEDCONSTANT._serialized_end=2267
  _SAVEDVARIABLE._serialized_start=2270
  _SAVEDVARIABLE._serialized_end=2618
  _FUNCTIONSPEC._serialized_start=2621
  _FUNCTIONSPEC._serialized_end=2875
  _FUNCTIONSPEC_JITCOMPILE._serialized_start=2821
  _FUNCTIONSPEC_JITCOMPILE._serialized_end=2863
  _SAVEDRESOURCE._serialized_start=2877
  _SAVEDRESOURCE._serialized_end=2908
  _SAVEABLEOBJECT._serialized_start=2910
  _SAVEABLEOBJECT._serialized_end=2975
# @@protoc_insertion_point(module_scope)

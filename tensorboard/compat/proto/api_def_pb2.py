# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorboard/compat/proto/api_def.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tensorboard.compat.proto import attr_value_pb2 as tensorboard_dot_compat_dot_proto_dot_attr__value__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n&tensorboard/compat/proto/api_def.proto\x12\x0btensorboard\x1a)tensorboard/compat/proto/attr_value.proto\"\xe7\x05\n\x06\x41piDef\x12\x15\n\rgraph_op_name\x18\x01 \x01(\t\x12\x1b\n\x13\x64\x65precation_message\x18\x0c \x01(\t\x12\x1b\n\x13\x64\x65precation_version\x18\r \x01(\x05\x12\x32\n\nvisibility\x18\x02 \x01(\x0e\x32\x1e.tensorboard.ApiDef.Visibility\x12.\n\x08\x65ndpoint\x18\x03 \x03(\x0b\x32\x1c.tensorboard.ApiDef.Endpoint\x12\'\n\x06in_arg\x18\x04 \x03(\x0b\x32\x17.tensorboard.ApiDef.Arg\x12(\n\x07out_arg\x18\x05 \x03(\x0b\x32\x17.tensorboard.ApiDef.Arg\x12\x11\n\targ_order\x18\x0b \x03(\t\x12&\n\x04\x61ttr\x18\x06 \x03(\x0b\x32\x18.tensorboard.ApiDef.Attr\x12\x0f\n\x07summary\x18\x07 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x08 \x01(\t\x12\x1a\n\x12\x64\x65scription_prefix\x18\t \x01(\t\x12\x1a\n\x12\x64\x65scription_suffix\x18\n \x01(\t\x1aI\n\x08\x45ndpoint\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\ndeprecated\x18\x03 \x01(\x08\x12\x1b\n\x13\x64\x65precation_version\x18\x04 \x01(\x05\x1a;\n\x03\x41rg\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\trename_to\x18\x02 \x01(\t\x12\x13\n\x0b\x64\x65scription\x18\x03 \x01(\t\x1ak\n\x04\x41ttr\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x11\n\trename_to\x18\x02 \x01(\t\x12-\n\rdefault_value\x18\x03 \x01(\x0b\x32\x16.tensorboard.AttrValue\x12\x13\n\x0b\x64\x65scription\x18\x04 \x01(\t\"G\n\nVisibility\x12\x16\n\x12\x44\x45\x46\x41ULT_VISIBILITY\x10\x00\x12\x0b\n\x07VISIBLE\x10\x01\x12\x08\n\x04SKIP\x10\x02\x12\n\n\x06HIDDEN\x10\x03\"*\n\x07\x41piDefs\x12\x1f\n\x02op\x18\x01 \x03(\x0b\x32\x13.tensorboard.ApiDefB}\n\x18org.tensorflow.frameworkB\x0c\x41piDefProtosP\x01ZNgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/api_def_go_proto\xf8\x01\x01\x62\x06proto3')



_APIDEF = DESCRIPTOR.message_types_by_name['ApiDef']
_APIDEF_ENDPOINT = _APIDEF.nested_types_by_name['Endpoint']
_APIDEF_ARG = _APIDEF.nested_types_by_name['Arg']
_APIDEF_ATTR = _APIDEF.nested_types_by_name['Attr']
_APIDEFS = DESCRIPTOR.message_types_by_name['ApiDefs']
_APIDEF_VISIBILITY = _APIDEF.enum_types_by_name['Visibility']
ApiDef = _reflection.GeneratedProtocolMessageType('ApiDef', (_message.Message,), {

  'Endpoint' : _reflection.GeneratedProtocolMessageType('Endpoint', (_message.Message,), {
    'DESCRIPTOR' : _APIDEF_ENDPOINT,
    '__module__' : 'tensorboard.compat.proto.api_def_pb2'
    # @@protoc_insertion_point(class_scope:tensorboard.ApiDef.Endpoint)
    })
  ,

  'Arg' : _reflection.GeneratedProtocolMessageType('Arg', (_message.Message,), {
    'DESCRIPTOR' : _APIDEF_ARG,
    '__module__' : 'tensorboard.compat.proto.api_def_pb2'
    # @@protoc_insertion_point(class_scope:tensorboard.ApiDef.Arg)
    })
  ,

  'Attr' : _reflection.GeneratedProtocolMessageType('Attr', (_message.Message,), {
    'DESCRIPTOR' : _APIDEF_ATTR,
    '__module__' : 'tensorboard.compat.proto.api_def_pb2'
    # @@protoc_insertion_point(class_scope:tensorboard.ApiDef.Attr)
    })
  ,
  'DESCRIPTOR' : _APIDEF,
  '__module__' : 'tensorboard.compat.proto.api_def_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.ApiDef)
  })
_sym_db.RegisterMessage(ApiDef)
_sym_db.RegisterMessage(ApiDef.Endpoint)
_sym_db.RegisterMessage(ApiDef.Arg)
_sym_db.RegisterMessage(ApiDef.Attr)

ApiDefs = _reflection.GeneratedProtocolMessageType('ApiDefs', (_message.Message,), {
  'DESCRIPTOR' : _APIDEFS,
  '__module__' : 'tensorboard.compat.proto.api_def_pb2'
  # @@protoc_insertion_point(class_scope:tensorboard.ApiDefs)
  })
_sym_db.RegisterMessage(ApiDefs)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\030org.tensorflow.frameworkB\014ApiDefProtosP\001ZNgithub.com/tensorflow/tensorflow/tensorflow/go/core/framework/api_def_go_proto\370\001\001'
  _APIDEF._serialized_start=99
  _APIDEF._serialized_end=842
  _APIDEF_ENDPOINT._serialized_start=526
  _APIDEF_ENDPOINT._serialized_end=599
  _APIDEF_ARG._serialized_start=601
  _APIDEF_ARG._serialized_end=660
  _APIDEF_ATTR._serialized_start=662
  _APIDEF_ATTR._serialized_end=769
  _APIDEF_VISIBILITY._serialized_start=771
  _APIDEF_VISIBILITY._serialized_end=842
  _APIDEFS._serialized_start=844
  _APIDEFS._serialized_end=886
# @@protoc_insertion_point(module_scope)

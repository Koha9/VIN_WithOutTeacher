ó
Í£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18ÿ
w
	h0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	h0/kernel
p
h0/kernel/Read/ReadVariableOpReadVariableOp	h0/kernel*'
_output_shapes
:*
dtype0
g
h0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	h0/bias
`
h0/bias/Read/ReadVariableOpReadVariableOph0/bias*
_output_shapes	
:*
dtype0
u
r/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
r/kernel
n
r/kernel/Read/ReadVariableOpReadVariableOpr/kernel*'
_output_shapes
:*
dtype0
t
q/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
q/kernel
m
q/kernel/Read/ReadVariableOpReadVariableOpq/kernel*&
_output_shapes
:
*
dtype0
x

q/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
q/kernel_1
q
q/kernel_1/Read/ReadVariableOpReadVariableOp
q/kernel_1*&
_output_shapes
:
*
dtype0
x

q/kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_name
q/kernel_2
q
q/kernel_2/Read/ReadVariableOpReadVariableOp
q/kernel_2*&
_output_shapes
:
*
dtype0
v
logits/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*
shared_namelogits/kernel
o
!logits/kernel/Read/ReadVariableOpReadVariableOplogits/kernel*
_output_shapes

:
*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ó
valueÉBÆ B¿
{
	conv0
	conv1
	conv2
conv_ShareWeights
conv_ShareWeights2

dense0
	keras_api

signatures
h

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
^

kernel
regularization_losses
trainable_variables
	variables
	keras_api
^

kernel
regularization_losses
trainable_variables
	variables
	keras_api
^

kernel
regularization_losses
trainable_variables
	variables
	keras_api
^

kernel
regularization_losses
 trainable_variables
!	variables
"	keras_api
^

#kernel
$regularization_losses
%trainable_variables
&	variables
'	keras_api
 
 
FD
VARIABLE_VALUE	h0/kernel'conv0/kernel/.ATTRIBUTES/VARIABLE_VALUE
B@
VARIABLE_VALUEh0/bias%conv0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

	0

1

	0

1
­
regularization_losses

(layers
)non_trainable_variables
trainable_variables
	variables
*layer_metrics
+metrics
,layer_regularization_losses
EC
VARIABLE_VALUEr/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
regularization_losses

-layers
.non_trainable_variables
trainable_variables
	variables
/layer_metrics
0metrics
1layer_regularization_losses
EC
VARIABLE_VALUEq/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
regularization_losses

2layers
3non_trainable_variables
trainable_variables
	variables
4layer_metrics
5metrics
6layer_regularization_losses
SQ
VARIABLE_VALUE
q/kernel_13conv_ShareWeights/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
regularization_losses

7layers
8non_trainable_variables
trainable_variables
	variables
9layer_metrics
:metrics
;layer_regularization_losses
TR
VARIABLE_VALUE
q/kernel_24conv_ShareWeights2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
­
regularization_losses

<layers
=non_trainable_variables
 trainable_variables
!	variables
>layer_metrics
?metrics
@layer_regularization_losses
KI
VARIABLE_VALUElogits/kernel(dense0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

#0

#0
­
$regularization_losses

Alayers
Bnon_trainable_variables
%trainable_variables
&	variables
Clayer_metrics
Dmetrics
Elayer_regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCallStatefulPartitionedCallsaver_filenameh0/kernel/Read/ReadVariableOph0/bias/Read/ReadVariableOpr/kernel/Read/ReadVariableOpq/kernel/Read/ReadVariableOpq/kernel_1/Read/ReadVariableOpq/kernel_2/Read/ReadVariableOp!logits/kernel/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *)
f$R"
 __inference__traced_save_4973385
ò
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename	h0/kernelh0/biasr/kernelq/kernel
q/kernel_1
q/kernel_2logits/kernel*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *,
f'R%
#__inference__traced_restore_4973416ÅÖ


 __inference__traced_save_4973385
file_prefix(
$savev2_h0_kernel_read_readvariableop&
"savev2_h0_bias_read_readvariableop'
#savev2_r_kernel_read_readvariableop'
#savev2_q_kernel_read_readvariableop)
%savev2_q_kernel_1_read_readvariableop)
%savev2_q_kernel_2_read_readvariableop,
(savev2_logits_kernel_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_c15628737f8242b0a7de6db8e9fd1981/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameØ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ê
valueàBÝB'conv0/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv0/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB3conv_ShareWeights/kernel/.ATTRIBUTES/VARIABLE_VALUEB4conv_ShareWeights2/kernel/.ATTRIBUTES/VARIABLE_VALUEB(dense0/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
SaveV2/shape_and_slicesÍ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_h0_kernel_read_readvariableop"savev2_h0_bias_read_readvariableop#savev2_r_kernel_read_readvariableop#savev2_q_kernel_read_readvariableop%savev2_q_kernel_1_read_readvariableop%savev2_q_kernel_2_read_readvariableop(savev2_logits_kernel_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapess
q: ::::
:
:
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
::!

_output_shapes	
::-)
'
_output_shapes
::,(
&
_output_shapes
:
:,(
&
_output_shapes
:
:,(
&
_output_shapes
:
:$ 

_output_shapes

:
:

_output_shapes
: 
ò 
È
#__inference__traced_restore_4973416
file_prefix
assignvariableop_h0_kernel
assignvariableop_1_h0_bias
assignvariableop_2_r_kernel
assignvariableop_3_q_kernel!
assignvariableop_4_q_kernel_1!
assignvariableop_5_q_kernel_2$
 assignvariableop_6_logits_kernel

identity_8¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6Þ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ê
valueàBÝB'conv0/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv0/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB3conv_ShareWeights/kernel/.ATTRIBUTES/VARIABLE_VALUEB4conv_ShareWeights2/kernel/.ATTRIBUTES/VARIABLE_VALUEB(dense0/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B 2
RestoreV2/shape_and_slicesÓ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_h0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOpassignvariableop_1_h0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2 
AssignVariableOp_2AssignVariableOpassignvariableop_2_r_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3 
AssignVariableOp_3AssignVariableOpassignvariableop_3_q_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¢
AssignVariableOp_4AssignVariableOpassignvariableop_4_q_kernel_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¢
AssignVariableOp_5AssignVariableOpassignvariableop_5_q_kernel_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¥
AssignVariableOp_6AssignVariableOp assignvariableop_6_logits_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpù

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_7ë

Identity_8IdentityIdentity_7:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*
T0*
_output_shapes
: 2

Identity_8"!

identity_8Identity_8:output:0*1
_input_shapes 
: :::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
®
ý
__inference_call_167608

inputs
s1
s2%
!h0_conv2d_readvariableop_resource&
"h0_biasadd_readvariableop_resource$
 r_conv2d_readvariableop_resource$
 q_conv2d_readvariableop_resource&
"q_conv2d_1_readvariableop_resource'
#q_conv2d_36_readvariableop_resource)
%logits_matmul_readvariableop_resource
identity

identity_1

identity_2
h0/Conv2D/ReadVariableOpReadVariableOp!h0_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
h0/Conv2D/ReadVariableOp¥
	h0/Conv2DConv2Dinputs h0/Conv2D/ReadVariableOp:value:0*
T0*(
_output_shapes
:*
paddingSAME*
strides
2
	h0/Conv2D
h0/BiasAdd/ReadVariableOpReadVariableOp"h0_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
h0/BiasAdd/ReadVariableOp

h0/BiasAddBiasAddh0/Conv2D:output:0!h0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:2

h0/BiasAdd
r/Conv2D/ReadVariableOpReadVariableOp r_conv2d_readvariableop_resource*'
_output_shapes
:*
dtype02
r/Conv2D/ReadVariableOp®
r/Conv2DConv2Dh0/BiasAdd:output:0r/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2

r/Conv2D
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"            2
zeros_like/shape_as_tensori
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_like/Const

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*'
_output_shapes
:2

zeros_like\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2r/Conv2D:output:0zeros_like:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:2
concat
q/Conv2D/ReadVariableOpReadVariableOp q_conv2d_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D/ReadVariableOpª
q/Conv2DConv2Dconcat:output:0q/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2

q/Conv2Dl
v/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v/reduction_indices
vMaxq/Conv2D:output:0v/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v`
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_1/axis
concat_1ConcatV2r/Conv2D:output:0
v:output:0concat_1/axis:output:0*
N*
T0*'
_output_shapes
:2

concat_1¡
q/Conv2D_1/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_1/ReadVariableOp²

q/Conv2D_1Conv2Dconcat_1:output:0!q/Conv2D_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2

q/Conv2D_1p
v_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_1/reduction_indices
v_1Maxq/Conv2D_1:output:0v_1/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_1`
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_2/axis
concat_2ConcatV2r/Conv2D:output:0v_1:output:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:2

concat_2¡
q/Conv2D_2/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_2/ReadVariableOp²

q/Conv2D_2Conv2Dconcat_2:output:0!q/Conv2D_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2

q/Conv2D_2p
v_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_2/reduction_indices
v_2Maxq/Conv2D_2:output:0v_2/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_2`
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_3/axis
concat_3ConcatV2r/Conv2D:output:0v_2:output:0concat_3/axis:output:0*
N*
T0*'
_output_shapes
:2

concat_3¡
q/Conv2D_3/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_3/ReadVariableOp²

q/Conv2D_3Conv2Dconcat_3:output:0!q/Conv2D_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2

q/Conv2D_3p
v_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_3/reduction_indices
v_3Maxq/Conv2D_3:output:0v_3/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_3`
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_4/axis
concat_4ConcatV2r/Conv2D:output:0v_3:output:0concat_4/axis:output:0*
N*
T0*'
_output_shapes
:2

concat_4¡
q/Conv2D_4/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_4/ReadVariableOp²

q/Conv2D_4Conv2Dconcat_4:output:0!q/Conv2D_4/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2

q/Conv2D_4p
v_4/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_4/reduction_indices
v_4Maxq/Conv2D_4:output:0v_4/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_4`
concat_5/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_5/axis
concat_5ConcatV2r/Conv2D:output:0v_4:output:0concat_5/axis:output:0*
N*
T0*'
_output_shapes
:2

concat_5¡
q/Conv2D_5/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_5/ReadVariableOp²

q/Conv2D_5Conv2Dconcat_5:output:0!q/Conv2D_5/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2

q/Conv2D_5p
v_5/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_5/reduction_indices
v_5Maxq/Conv2D_5:output:0v_5/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_5`
concat_6/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_6/axis
concat_6ConcatV2r/Conv2D:output:0v_5:output:0concat_6/axis:output:0*
N*
T0*'
_output_shapes
:2

concat_6¡
q/Conv2D_6/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_6/ReadVariableOp²

q/Conv2D_6Conv2Dconcat_6:output:0!q/Conv2D_6/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2

q/Conv2D_6p
v_6/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_6/reduction_indices
v_6Maxq/Conv2D_6:output:0v_6/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_6`
concat_7/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_7/axis
concat_7ConcatV2r/Conv2D:output:0v_6:output:0concat_7/axis:output:0*
N*
T0*'
_output_shapes
:2

concat_7¡
q/Conv2D_7/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_7/ReadVariableOp²

q/Conv2D_7Conv2Dconcat_7:output:0!q/Conv2D_7/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2

q/Conv2D_7p
v_7/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_7/reduction_indices
v_7Maxq/Conv2D_7:output:0v_7/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_7`
concat_8/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_8/axis
concat_8ConcatV2r/Conv2D:output:0v_7:output:0concat_8/axis:output:0*
N*
T0*'
_output_shapes
:2

concat_8¡
q/Conv2D_8/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_8/ReadVariableOp²

q/Conv2D_8Conv2Dconcat_8:output:0!q/Conv2D_8/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2

q/Conv2D_8p
v_8/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_8/reduction_indices
v_8Maxq/Conv2D_8:output:0v_8/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_8`
concat_9/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_9/axis
concat_9ConcatV2r/Conv2D:output:0v_8:output:0concat_9/axis:output:0*
N*
T0*'
_output_shapes
:2

concat_9¡
q/Conv2D_9/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_9/ReadVariableOp²

q/Conv2D_9Conv2Dconcat_9:output:0!q/Conv2D_9/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2

q/Conv2D_9p
v_9/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_9/reduction_indices
v_9Maxq/Conv2D_9:output:0v_9/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_9b
concat_10/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_10/axis
	concat_10ConcatV2r/Conv2D:output:0v_9:output:0concat_10/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_10£
q/Conv2D_10/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_10/ReadVariableOp¶
q/Conv2D_10Conv2Dconcat_10:output:0"q/Conv2D_10/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_10r
v_10/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_10/reduction_indices
v_10Maxq/Conv2D_10:output:0v_10/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_10b
concat_11/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_11/axis
	concat_11ConcatV2r/Conv2D:output:0v_10:output:0concat_11/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_11£
q/Conv2D_11/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_11/ReadVariableOp¶
q/Conv2D_11Conv2Dconcat_11:output:0"q/Conv2D_11/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_11r
v_11/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_11/reduction_indices
v_11Maxq/Conv2D_11:output:0v_11/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_11b
concat_12/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_12/axis
	concat_12ConcatV2r/Conv2D:output:0v_11:output:0concat_12/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_12£
q/Conv2D_12/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_12/ReadVariableOp¶
q/Conv2D_12Conv2Dconcat_12:output:0"q/Conv2D_12/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_12r
v_12/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_12/reduction_indices
v_12Maxq/Conv2D_12:output:0v_12/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_12b
concat_13/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_13/axis
	concat_13ConcatV2r/Conv2D:output:0v_12:output:0concat_13/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_13£
q/Conv2D_13/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_13/ReadVariableOp¶
q/Conv2D_13Conv2Dconcat_13:output:0"q/Conv2D_13/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_13r
v_13/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_13/reduction_indices
v_13Maxq/Conv2D_13:output:0v_13/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_13b
concat_14/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_14/axis
	concat_14ConcatV2r/Conv2D:output:0v_13:output:0concat_14/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_14£
q/Conv2D_14/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_14/ReadVariableOp¶
q/Conv2D_14Conv2Dconcat_14:output:0"q/Conv2D_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_14r
v_14/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_14/reduction_indices
v_14Maxq/Conv2D_14:output:0v_14/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_14b
concat_15/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_15/axis
	concat_15ConcatV2r/Conv2D:output:0v_14:output:0concat_15/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_15£
q/Conv2D_15/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_15/ReadVariableOp¶
q/Conv2D_15Conv2Dconcat_15:output:0"q/Conv2D_15/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_15r
v_15/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_15/reduction_indices
v_15Maxq/Conv2D_15:output:0v_15/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_15b
concat_16/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_16/axis
	concat_16ConcatV2r/Conv2D:output:0v_15:output:0concat_16/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_16£
q/Conv2D_16/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_16/ReadVariableOp¶
q/Conv2D_16Conv2Dconcat_16:output:0"q/Conv2D_16/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_16r
v_16/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_16/reduction_indices
v_16Maxq/Conv2D_16:output:0v_16/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_16b
concat_17/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_17/axis
	concat_17ConcatV2r/Conv2D:output:0v_16:output:0concat_17/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_17£
q/Conv2D_17/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_17/ReadVariableOp¶
q/Conv2D_17Conv2Dconcat_17:output:0"q/Conv2D_17/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_17r
v_17/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_17/reduction_indices
v_17Maxq/Conv2D_17:output:0v_17/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_17b
concat_18/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_18/axis
	concat_18ConcatV2r/Conv2D:output:0v_17:output:0concat_18/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_18£
q/Conv2D_18/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_18/ReadVariableOp¶
q/Conv2D_18Conv2Dconcat_18:output:0"q/Conv2D_18/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_18r
v_18/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_18/reduction_indices
v_18Maxq/Conv2D_18:output:0v_18/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_18b
concat_19/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_19/axis
	concat_19ConcatV2r/Conv2D:output:0v_18:output:0concat_19/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_19£
q/Conv2D_19/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_19/ReadVariableOp¶
q/Conv2D_19Conv2Dconcat_19:output:0"q/Conv2D_19/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_19r
v_19/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_19/reduction_indices
v_19Maxq/Conv2D_19:output:0v_19/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_19b
concat_20/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_20/axis
	concat_20ConcatV2r/Conv2D:output:0v_19:output:0concat_20/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_20£
q/Conv2D_20/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_20/ReadVariableOp¶
q/Conv2D_20Conv2Dconcat_20:output:0"q/Conv2D_20/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_20r
v_20/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_20/reduction_indices
v_20Maxq/Conv2D_20:output:0v_20/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_20b
concat_21/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_21/axis
	concat_21ConcatV2r/Conv2D:output:0v_20:output:0concat_21/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_21£
q/Conv2D_21/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_21/ReadVariableOp¶
q/Conv2D_21Conv2Dconcat_21:output:0"q/Conv2D_21/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_21r
v_21/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_21/reduction_indices
v_21Maxq/Conv2D_21:output:0v_21/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_21b
concat_22/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_22/axis
	concat_22ConcatV2r/Conv2D:output:0v_21:output:0concat_22/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_22£
q/Conv2D_22/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_22/ReadVariableOp¶
q/Conv2D_22Conv2Dconcat_22:output:0"q/Conv2D_22/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_22r
v_22/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_22/reduction_indices
v_22Maxq/Conv2D_22:output:0v_22/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_22b
concat_23/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_23/axis
	concat_23ConcatV2r/Conv2D:output:0v_22:output:0concat_23/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_23£
q/Conv2D_23/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_23/ReadVariableOp¶
q/Conv2D_23Conv2Dconcat_23:output:0"q/Conv2D_23/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_23r
v_23/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_23/reduction_indices
v_23Maxq/Conv2D_23:output:0v_23/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_23b
concat_24/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_24/axis
	concat_24ConcatV2r/Conv2D:output:0v_23:output:0concat_24/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_24£
q/Conv2D_24/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_24/ReadVariableOp¶
q/Conv2D_24Conv2Dconcat_24:output:0"q/Conv2D_24/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_24r
v_24/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_24/reduction_indices
v_24Maxq/Conv2D_24:output:0v_24/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_24b
concat_25/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_25/axis
	concat_25ConcatV2r/Conv2D:output:0v_24:output:0concat_25/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_25£
q/Conv2D_25/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_25/ReadVariableOp¶
q/Conv2D_25Conv2Dconcat_25:output:0"q/Conv2D_25/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_25r
v_25/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_25/reduction_indices
v_25Maxq/Conv2D_25:output:0v_25/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_25b
concat_26/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_26/axis
	concat_26ConcatV2r/Conv2D:output:0v_25:output:0concat_26/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_26£
q/Conv2D_26/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_26/ReadVariableOp¶
q/Conv2D_26Conv2Dconcat_26:output:0"q/Conv2D_26/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_26r
v_26/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_26/reduction_indices
v_26Maxq/Conv2D_26:output:0v_26/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_26b
concat_27/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_27/axis
	concat_27ConcatV2r/Conv2D:output:0v_26:output:0concat_27/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_27£
q/Conv2D_27/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_27/ReadVariableOp¶
q/Conv2D_27Conv2Dconcat_27:output:0"q/Conv2D_27/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_27r
v_27/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_27/reduction_indices
v_27Maxq/Conv2D_27:output:0v_27/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_27b
concat_28/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_28/axis
	concat_28ConcatV2r/Conv2D:output:0v_27:output:0concat_28/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_28£
q/Conv2D_28/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_28/ReadVariableOp¶
q/Conv2D_28Conv2Dconcat_28:output:0"q/Conv2D_28/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_28r
v_28/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_28/reduction_indices
v_28Maxq/Conv2D_28:output:0v_28/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_28b
concat_29/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_29/axis
	concat_29ConcatV2r/Conv2D:output:0v_28:output:0concat_29/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_29£
q/Conv2D_29/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_29/ReadVariableOp¶
q/Conv2D_29Conv2Dconcat_29:output:0"q/Conv2D_29/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_29r
v_29/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_29/reduction_indices
v_29Maxq/Conv2D_29:output:0v_29/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_29b
concat_30/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_30/axis
	concat_30ConcatV2r/Conv2D:output:0v_29:output:0concat_30/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_30£
q/Conv2D_30/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_30/ReadVariableOp¶
q/Conv2D_30Conv2Dconcat_30:output:0"q/Conv2D_30/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_30r
v_30/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_30/reduction_indices
v_30Maxq/Conv2D_30:output:0v_30/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_30b
concat_31/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_31/axis
	concat_31ConcatV2r/Conv2D:output:0v_30:output:0concat_31/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_31£
q/Conv2D_31/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_31/ReadVariableOp¶
q/Conv2D_31Conv2Dconcat_31:output:0"q/Conv2D_31/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_31r
v_31/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_31/reduction_indices
v_31Maxq/Conv2D_31:output:0v_31/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_31b
concat_32/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_32/axis
	concat_32ConcatV2r/Conv2D:output:0v_31:output:0concat_32/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_32£
q/Conv2D_32/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_32/ReadVariableOp¶
q/Conv2D_32Conv2Dconcat_32:output:0"q/Conv2D_32/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_32r
v_32/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_32/reduction_indices
v_32Maxq/Conv2D_32:output:0v_32/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_32b
concat_33/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_33/axis
	concat_33ConcatV2r/Conv2D:output:0v_32:output:0concat_33/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_33£
q/Conv2D_33/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_33/ReadVariableOp¶
q/Conv2D_33Conv2Dconcat_33:output:0"q/Conv2D_33/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_33r
v_33/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_33/reduction_indices
v_33Maxq/Conv2D_33:output:0v_33/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_33b
concat_34/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_34/axis
	concat_34ConcatV2r/Conv2D:output:0v_33:output:0concat_34/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_34£
q/Conv2D_34/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_34/ReadVariableOp¶
q/Conv2D_34Conv2Dconcat_34:output:0"q/Conv2D_34/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_34r
v_34/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_34/reduction_indices
v_34Maxq/Conv2D_34:output:0v_34/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_34b
concat_35/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_35/axis
	concat_35ConcatV2r/Conv2D:output:0v_34:output:0concat_35/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_35£
q/Conv2D_35/ReadVariableOpReadVariableOp"q_conv2d_1_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_35/ReadVariableOp¶
q/Conv2D_35Conv2Dconcat_35:output:0"q/Conv2D_35/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_35r
v_35/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
v_35/reduction_indices
v_35Maxq/Conv2D_35:output:0v_35/reduction_indices:output:0*
T0*'
_output_shapes
:*
	keep_dims(2
v_35b
concat_36/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat_36/axis
	concat_36ConcatV2r/Conv2D:output:0v_35:output:0concat_36/axis:output:0*
N*
T0*'
_output_shapes
:2
	concat_36¤
q/Conv2D_36/ReadVariableOpReadVariableOp#q_conv2d_36_readvariableop_resource*&
_output_shapes
:
*
dtype02
q/Conv2D_36/ReadVariableOp¶
q/Conv2D_36Conv2Dconcat_36:output:0"q/Conv2D_36/ReadVariableOp:value:0*
T0*'
_output_shapes
:
*
paddingSAME*
strides
2
q/Conv2D_36M
CastCasts1*

DstT0*

SrcT0*
_output_shapes	
:2
CastQ
Cast_1Casts2*

DstT0*

SrcT0*
_output_shapes	
:2
Cast_1g
ShapeConst*
_output_shapes
:*
dtype0*%
valueB"         
   2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltax
rangeRangerange/start:output:0strided_slice:output:0range/delta:output:0*
_output_shapes	
:2
range{
stackPackrange:output:0Cast:y:0
Cast_1:y:0*
N*
T0*
_output_shapes
:	*

axis2
stack
q_outGatherNdq/Conv2D_36:output:0stack:output:0*
Tindices0*
Tparams0*
_output_shapes
:	
2
q_out¢
logits/MatMul/ReadVariableOpReadVariableOp%logits_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02
logits/MatMul/ReadVariableOp
logits/MatMulMatMulq_out:output:0$logits/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
logits/MatMulx
probability_actionsSoftmaxlogits/MatMul:product:0*
T0*
_output_shapes
:	2
probability_actionsc
IdentityIdentitylogits/MatMul:product:0*
T0*
_output_shapes
:	2

Identitym

Identity_1Identityprobability_actions:softmax:0*
T0*
_output_shapes
:	2

Identity_1j

Identity_2Identityconcat_36:output:0*
T0*'
_output_shapes
:2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*P
_input_shapes?
=:::::::::::O K
'
_output_shapes
:
 
_user_specified_nameinputs:?;

_output_shapes	
:

_user_specified_nameS1:?;

_output_shapes	
:

_user_specified_nameS2"¸J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:n
ó
	conv0
	conv1
	conv2
conv_ShareWeights
conv_ShareWeights2

dense0
	keras_api

signatures
Fcall"ë
_tf_keras_modelÑ{"class_name": "VIN", "name": "vin", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "VIN"}}



	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
*G&call_and_return_all_conditional_losses
H__call__"Û
_tf_keras_layerÁ{"class_name": "Conv2D", "name": "h0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "h0", "trainable": true, "dtype": "float32", "filters": 150, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 8, 8, 2]}}
Ö	

kernel
regularization_losses
trainable_variables
	variables
	keras_api
*I&call_and_return_all_conditional_losses
J__call__"»
_tf_keras_layer¡{"class_name": "Conv2D", "name": "r", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "r", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "bias_initializer": null, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 150}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 8, 8, 150]}}
Ó	

kernel
regularization_losses
trainable_variables
	variables
	keras_api
*K&call_and_return_all_conditional_losses
L__call__"¸
_tf_keras_layer{"class_name": "Conv2D", "name": "q", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "q", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "bias_initializer": null, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 8, 8, 2]}}
Ó	

kernel
regularization_losses
trainable_variables
	variables
	keras_api
*M&call_and_return_all_conditional_losses
N__call__"¸
_tf_keras_layer{"class_name": "Conv2D", "name": "q", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "q", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "bias_initializer": null, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 8, 8, 2]}}
Ó	

kernel
regularization_losses
 trainable_variables
!	variables
"	keras_api
*O&call_and_return_all_conditional_losses
P__call__"¸
_tf_keras_layer{"class_name": "Conv2D", "name": "q", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "q", "trainable": true, "dtype": "float32", "filters": 10, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "bias_initializer": null, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 8, 8, 2]}}


#kernel
$regularization_losses
%trainable_variables
&	variables
'	keras_api
*Q&call_and_return_all_conditional_losses
R__call__"æ
_tf_keras_layerÌ{"class_name": "Dense", "name": "logits", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "logits", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.01, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [128, 10]}}
"
_generic_user_object
"
signature_map
$:"2	h0/kernel
:2h0/bias
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
­
regularization_losses

(layers
)non_trainable_variables
trainable_variables
	variables
*layer_metrics
+metrics
,layer_regularization_losses
H__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
#:!2r/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
regularization_losses

-layers
.non_trainable_variables
trainable_variables
	variables
/layer_metrics
0metrics
1layer_regularization_losses
J__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
": 
2q/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
regularization_losses

2layers
3non_trainable_variables
trainable_variables
	variables
4layer_metrics
5metrics
6layer_regularization_losses
L__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
": 
2q/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
regularization_losses

7layers
8non_trainable_variables
trainable_variables
	variables
9layer_metrics
:metrics
;layer_regularization_losses
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
": 
2q/kernel
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
­
regularization_losses

<layers
=non_trainable_variables
 trainable_variables
!	variables
>layer_metrics
?metrics
@layer_regularization_losses
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
:
2logits/kernel
 "
trackable_list_wrapper
'
#0"
trackable_list_wrapper
'
#0"
trackable_list_wrapper
­
$regularization_losses

Alayers
Bnon_trainable_variables
%trainable_variables
&	variables
Clayer_metrics
Dmetrics
Elayer_regularization_losses
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Û2Ø
__inference_call_167608¼
³²¯
FullArgSpec2
args*'
jself
jinputs
jS1
jS2
jVInum
varargs
 
varkw
 
defaults¢
`

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 Ê
__inference_call_167608®	
#W¢T
M¢J
 
inputs

S1

S2
`H
ª "J¢G

0	

1	

2
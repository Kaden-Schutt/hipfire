; ModuleID = 'aie_kernels/generic/expand.cc'
source_filename = "aie_kernels/generic/expand.cc"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p-none-unknown-elf"

; Function Attrs: mustprogress nofree nosync nounwind
define dso_local void @expand_int4_to_bfloat16(ptr readonly captures(none) %a_in, ptr writeonly captures(none) %c_out) local_unnamed_addr #0 {
entry:
  %add.ptr.i = getelementptr inbounds nuw i8, ptr %a_in, i20 4096
  tail call void @llvm.aie2p.event(i32 0)
  %0 = tail call noundef <32 x float> @llvm.aie2p.v32bf16.to.v32accfloat(<32 x bfloat> splat (bfloat 0xR5301))
  %1 = tail call noundef <32 x float> @llvm.aie2p.v32bf16.to.v32accfloat(<32 x bfloat> splat (bfloat 0xR4B01))
  %2 = bitcast <32 x float> %1 to <16 x i64>
  %shuffle.i.i.i.i.i6.i.i.i.i.i.i.i = shufflevector <16 x i64> %2, <16 x i64> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %3 = bitcast <32 x i64> %shuffle.i.i.i.i.i6.i.i.i.i.i.i.i to <64 x float>
  br label %for.body.i

for.body.i:                                       ; preds = %for.cond.cleanup4.i, %entry
  %i.027.i = phi i32 [ 0, %entry ], [ %inc10.i, %for.cond.cleanup4.i ]
  %pO.026.i = phi ptr [ %c_out, %entry ], [ %add.ptr8.i, %for.cond.cleanup4.i ]
  %pSF.025.i = phi ptr [ %add.ptr.i, %entry ], [ %add.ptr1.i, %for.cond.cleanup4.i ]
  %pI.024.i = phi ptr [ %a_in, %entry ], [ %add.ptr6.i, %for.cond.cleanup4.i ]
  %4 = load bfloat, ptr %pSF.025.i, align 2, !tbaa !2
  %5 = insertelement <32 x bfloat> poison, bfloat %4, i64 0
  %6 = shufflevector <32 x bfloat> %5, <32 x bfloat> poison, <32 x i32> zeroinitializer
  br label %for.body5.i

for.cond.cleanup4.i:                              ; preds = %for.body5.i
  %add.ptr1.i = getelementptr inbounds nuw i8, ptr %pSF.025.i, i20 2
  %inc10.i = add nuw nsw i32 %i.027.i, 1
  %exitcond28.not.i = icmp eq i32 %inc10.i, 32
  br i1 %exitcond28.not.i, label %_Z6expandIDU4_8bfloat16S1_Li8192ELi256EEvPT_PT1_.exit, label %for.body.i, !llvm.loop !6

for.body5.i:                                      ; preds = %for.body5.i, %for.body.i
  %k.023.i = phi i32 [ 0, %for.body.i ], [ %inc.i, %for.body5.i ]
  %pO.122.i = phi ptr [ %pO.026.i, %for.body.i ], [ %add.ptr8.i, %for.body5.i ]
  %pI.121.i = phi ptr [ %pI.024.i, %for.body.i ], [ %add.ptr6.i, %for.body5.i ]
  %7 = load <4 x i32>, ptr %pI.121.i, align 16, !tbaa !8
  %add.ptr6.i = getelementptr inbounds nuw i8, ptr %pI.121.i, i20 16
  %shuffle.i.i.i.i.i.i.i.i.i.i = shufflevector <4 x i32> %7, <4 x i32> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %8 = bitcast <8 x i32> %shuffle.i.i.i.i.i.i.i.i.i.i to <32 x i8>
  %9 = tail call noundef <64 x i8> @llvm.aie2p.unpack.I512.I8.I4(<32 x i8> %8, i32 0)
  %10 = bitcast <64 x i8> %9 to <16 x i32>
  %shuffle.i.i.i.i.i.i.i.i.i.i.i = shufflevector <16 x i32> %10, <16 x i32> poison, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %11 = bitcast <8 x i32> %shuffle.i.i.i.i.i.i.i.i.i.i.i to <32 x i8>
  %12 = tail call noundef <32 x i16> @llvm.aie2p.unpack.I512.I16.I8(<32 x i8> %11, i32 0)
  %13 = tail call noundef <32 x i32> @llvm.aie2p.acc32.v32.I512.ups(<32 x i16> %12, i32 0, i32 range(i32 0, 2) 0)
  %14 = bitcast <32 x i32> %13 to <16 x i64>
  %shuffle.i.i.i.i.i.i.i.i.i.i.i.i = shufflevector <16 x i64> %14, <16 x i64> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %15 = tail call <32 x i64> @llvm.aie2p.ACC2048.add.conf(<32 x i64> %shuffle.i.i.i.i.i.i.i.i.i.i.i.i, <32 x i64> %shuffle.i.i.i.i.i6.i.i.i.i.i.i.i, i32 0)
  %shuffle.i.i.i.i.i.i.i.i.i.i19.i = shufflevector <32 x i64> %15, <32 x i64> poison, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %16 = bitcast <32 x i64> %shuffle.i.i.i.i.i.i.i.i.i.i19.i to <64 x float>
  %17 = tail call noundef <64 x float> @llvm.aie2p.ACC2048.accfloat.sub.conf(<64 x float> %16, <64 x float> %3, i32 60)
  %18 = bitcast <64 x float> %17 to <32 x i64>
  %shuffle.i.i5.i.i.i.i.i.i.i.i.i = shufflevector <32 x i64> %18, <32 x i64> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %19 = bitcast <16 x i64> %shuffle.i.i5.i.i.i.i.i.i.i.i.i to <32 x float>
  %20 = tail call noundef <32 x bfloat> @llvm.aie2p.v32accfloat.to.v32bf16(<32 x float> %19)
  %21 = tail call noundef <32 x float> @llvm.aie2p.I512.I512.ACC1024.bf.mul.conf(<32 x bfloat> %20, <32 x bfloat> %6, i32 828)
  %22 = tail call noundef <32 x bfloat> @llvm.aie2p.v32accfloat.to.v32bf16(<32 x float> %21)
  store <32 x bfloat> %22, ptr %pO.122.i, align 64, !tbaa !8
  %add.ptr8.i = getelementptr inbounds nuw i8, ptr %pO.122.i, i20 64
  %inc.i = add nuw nsw i32 %k.023.i, 1
  %exitcond.not.i = icmp eq i32 %inc.i, 8
  br i1 %exitcond.not.i, label %for.cond.cleanup4.i, label %for.body5.i, !llvm.loop !9

_Z6expandIDU4_8bfloat16S1_Li8192ELi256EEvPT_PT1_.exit: ; preds = %for.cond.cleanup4.i
  tail call void @llvm.aie2p.event(i32 1)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.aie2p.event(i32) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <64 x i8> @llvm.aie2p.unpack.I512.I8.I4(<32 x i8>, i32) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i16> @llvm.aie2p.unpack.I512.I16.I8(<32 x i8>, i32) #2

; Function Attrs: nofree nosync nounwind memory(none)
declare <32 x float> @llvm.aie2p.v32bf16.to.v32accfloat(<32 x bfloat>) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare <32 x i32> @llvm.aie2p.acc32.v32.I512.ups(<32 x i16>, i32, i32) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(none)
declare <32 x i64> @llvm.aie2p.ACC2048.add.conf(<32 x i64>, <32 x i64>, i32) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare <64 x float> @llvm.aie2p.ACC2048.accfloat.sub.conf(<64 x float>, <64 x float>, i32) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare <32 x bfloat> @llvm.aie2p.v32accfloat.to.v32bf16(<32 x float>) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read)
declare <32 x float> @llvm.aie2p.I512.I512.ACC1024.bf.mul.conf(<32 x bfloat>, <32 x bfloat>, i32) #4

attributes #0 = { mustprogress nofree nosync nounwind "no-builtin-memcpy" "no-builtin-memmove" "no-builtin-memset" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nofree nosync nounwind memory(none) }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: read) }

!llvm.linker.options = !{}
!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 21.0.0 (https://github.com/Xilinx/llvm-aie 7820460ea745ca8634db6fb8093c00e0df5ded2d)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"__bf16", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!4, !4, i64 0}
!9 = distinct !{!9, !7}

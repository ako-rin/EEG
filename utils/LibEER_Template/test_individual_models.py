#!/usr/bin/env python3
"""
逐个测试每个模型的导入，找出具体的问题
"""

models_to_test = [
    'DGCNN', 'EEGNet', 'STRNN', 'GCBNet', 'DBN', 'TSception', 
    'SVM', 'CDCN', 'HSLT', 'ACRNN', 'GCBNet_BLS', 'MsMda'
]

print("🧪 逐个测试模型导入...")

successful_models = []
failed_models = []

for model_name in models_to_test:
    try:
        module = __import__(f'LibEER.models.{model_name}', fromlist=[model_name])
        # 特殊处理 MsMda 模块的类名
        if model_name == 'MsMda':
            model_class = getattr(module, 'MSMDA')
        else:
            model_class = getattr(module, model_name)
        successful_models.append(model_name)
        print(f"✅ {model_name}")
    except Exception as e:
        failed_models.append((model_name, str(e)))
        print(f"❌ {model_name}: {e}")

print(f"\n📊 测试结果:")
print(f"✅ 成功: {len(successful_models)} 个模型")
print(f"❌ 失败: {len(failed_models)} 个模型")

if failed_models:
    print(f"\n⚠️  失败的模型:")
    for model, error in failed_models:
        print(f"   {model}: {error}")

# 测试特殊的 RGNN_official
print(f"\n🔍 测试 RGNN_official:")
try:
    from LibEER.models.RGNN_official import SymSimGCNNet
    print("✅ RGNN_official 导入成功")
    successful_models.append('RGNN_official')
except Exception as e:
    print(f"❌ RGNN_official: {e}")
    failed_models.append(('RGNN_official', str(e)))

print(f"\n🎯 最终统计:")
print(f"✅ 成功模型数: {len(successful_models)}")
print(f"❌ 失败模型数: {len(failed_models)}")

if len(failed_models) == 0:
    print(f"\n🎉 所有模型导入成功！可以正常使用 LibEER 了！")
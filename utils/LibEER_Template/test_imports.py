#!/usr/bin/env python3
"""
测试 LibEER 包的所有核心导入是否正常
"""

def test_all_imports():
    try:
        print("🧪 测试 LibEER 包导入...")
        
        # 测试模型导入
        from LibEER.models.Models import Model
        print("✅ Models 导入成功")
        print(f"   可用模型: {list(Model.keys())[:5]}...")
        
        # 测试配置导入
        from LibEER.config.setting import Setting, preset_setting
        print("✅ Settings 导入成功")
        
        # 测试数据工具导入
        from LibEER.data_utils.load_data import get_data
        print("✅ Data utils 导入成功")
        
        # 测试常量导入
        from LibEER.data_utils.constants.seed import SEED_RGNN_ADJACENCY_MATRIX
        from LibEER.data_utils.constants.deap import DEAP_CHANNEL_NAME
        print("✅ Constants 导入成功")
        
        # 测试工具导入
        from LibEER.utils.args import get_args_parser
        print("✅ Utils 导入成功")
        
        # 测试训练器导入
        from LibEER.Trainer.training import train
        print("✅ Trainer 导入成功")
        
        print("\n🎉 所有核心模块导入成功！LibEER 包已修复完成！")
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_imports()
    if success:
        print("\n💡 现在你可以正常使用 LibEER 了！")
        print("   运行命令: python /home/ako/Project/work/ANN/LibEER_Template/main.py --help")
    else:
        print("\n⚠️  还有一些导入问题需要解决")